"""A história do treino: o que aconteceu num run, em capítulos com texto e visuais (ADR 0047)."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml_lab.config import Settings, get_settings
from ml_lab.core.data.io.base import SourceSpec
from ml_lab.core.data.preparation import PreparationConfig
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.foundation.types import Partition, TaskType
from ml_lab.core.schemas.experiment import ExperimentConfig, ModelSpec
from ml_lab.core.schemas.reinforcement import ReinforcementConfig
from ml_lab.core.services.experiments import ExperimentResult
from ml_lab.core.services.lab import Lab
from ml_lab.core.services.story import MAX_CURVE_POINTS, TrainingStory, training_story
from ml_lab.core.tracking.store import InMemoryStore
from ml_lab.core.training.recorder import StepRecord


def settings(folder: Path) -> Settings:
    return replace(
        get_settings(), artifacts_dir=folder / "artifacts", database_path=folder / "registo.db"
    )


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return Lab.create(settings(tmp_path), store=InMemoryStore())


def run(
    lab: Lab,
    sample: str = "iris",
    target: str | None = "species",
    model: str = "logistic_regression",
    *,
    task_type: TaskType | None = None,
    split: SplitConfig | None = None,
    **params: Any,
) -> ExperimentResult:
    source = lab.datasets.load_sample(sample).source
    preparation = PreparationConfig(
        target=target, task_type=task_type, split=split or SplitConfig()
    )
    config = ExperimentConfig(
        dataset=source,
        preparation=preparation,
        model=ModelSpec(name=model, hyperparameters=params),
        seed=7,
    )
    return lab.experiments.run(config)


def chapter_keys(story: TrainingStory) -> list[str]:
    return [chapter.key for chapter in story.chapters]


def test_a_classification_is_told_from_the_data_to_what_to_do_next(lab: Lab) -> None:
    result = run(lab, split=SplitConfig(SplitStrategy.STRATIFIED))

    story = lab.history.story(result.run_id)

    assert chapter_keys(story) == ["data", "learning", "learned", "result", "errors", "next"]
    assert "Logistic regression learned to predict species" in story.headline
    assert result.main_report.find("accuracy") is not None
    data, _, learned, outcome, errors, _ = story.chapters
    # As linhas: o treino e o teste, com o que cada parte faz.
    assert [part.partition for part in data.split] == [Partition.TRAIN, Partition.TEST]
    assert sum(part.rows for part in data.split) == 150
    # A preparação, com os números concretos (o run ainda está em memória).
    assert any("standardised" in point for point in data.points)
    # Os pesos de um modelo linear, e a comparação com o palpite mais simples.
    assert learned.bars is not None
    assert outcome.bars is not None
    assert [bar.tone for bar in outcome.bars.bars] == ["baseline", "train", "test"]
    assert "Always guessing" in outcome.text
    assert errors.matrix == result.main_report.confusion_matrix


def test_an_iterative_model_shows_its_loss_curve(lab: Lab) -> None:
    result = run(lab, model="logistic_regression_gd")

    learning = lab.history.story(result.run_id).chapters[1]

    assert learning.curve is not None
    assert learning.curve.step_label == "Epoch"
    assert learning.curve.steps[-1] == result.steps
    assert "The loss" in learning.text


def test_a_long_training_is_summed_up_in_a_few_points(lab: Lab) -> None:
    result = run(lab, model="logistic_regression_gd")
    history = tuple(StepRecord(step, 1.0 / step, {}) for step in range(1, 2_001))

    story = training_story(replace(result, history=history))

    curve = story.chapters[1].curve
    assert curve is not None
    assert len(curve.steps) <= MAX_CURVE_POINTS
    assert curve.steps[0] == 1
    assert curve.steps[-1] == 2_000


def test_a_model_without_steps_learns_in_one_go(lab: Lab) -> None:
    result = run(lab, model="knn")

    story = lab.history.story(result.run_id)

    learning, learned = story.chapters[1], story.chapters[2]
    assert learning.curve is None
    assert "in one go" in learning.text
    assert "training rows" in learned.text  # o KNN guarda as linhas de treino


def test_a_regression_says_how_big_the_errors_are(lab: Lab) -> None:
    result = run(
        lab,
        sample="california_housing",
        target="median_house_value",
        model="linear_regression",
    )

    story = lab.history.story(result.run_id)

    outcome = next(chapter for chapter in story.chapters if chapter.key == "result")
    errors = next(chapter for chapter in story.chapters if chapter.key == "errors")
    assert "R²" in outcome.text
    assert outcome.bars is not None
    assert outcome.bars.bars[0].tone == "baseline"  # sempre a média, que dá um R² de 0
    assert errors.bars is not None
    assert {bar.label for bar in errors.bars.bars} == {
        "MAE (the typical error)",
        "RMSE (big errors weigh more)",
    }


def test_a_clustering_keeps_the_reference_out_of_the_training(lab: Lab) -> None:
    result = run(lab, model="kmeans", task_type=TaskType.CLUSTERING)

    story = lab.history.story(result.run_id)

    data = story.chapters[0]
    assert "no target" in data.text
    assert "**species** stays out of the training" in data.text
    assert "split the rows into" in story.headline
    errors = next(chapter for chapter in story.chapters if chapter.key == "errors")
    assert errors.matrix_rows == "Reference class"


def test_the_warnings_become_what_to_try_next(lab: Lab) -> None:
    result = run(lab)
    warning = Notice.warning("overfitting", "The model memorised the training data.")

    story = training_story(replace(result, notices=(warning, warning)))

    last = story.chapters[-1]
    assert last.key == "next"
    assert last.notices == (warning,)  # sem repetir
    assert last.text.startswith("One thing to look at")


def test_a_detector_that_flags_nothing_is_told_so(lab: Lab) -> None:
    source = SourceSpec("synthetic", {"generator": "anomalies", "seed": 7, "parameters": {}})
    preparation = PreparationConfig(
        target="target",
        task_type=TaskType.ANOMALY_DETECTION,
        split=SplitConfig(SplitStrategy.STRATIFIED),  # o teste fica com as suas anomalias
    )
    # O limiar mais alto que há: no teste, nada parece anómalo.
    model = ModelSpec(name="isolation_forest", hyperparameters={"contamination": 0.001})
    result = lab.experiments.run(
        ExperimentConfig(dataset=source, preparation=preparation, model=model, seed=7)
    )

    story = lab.history.story(result.run_id)

    assert "flagged none of the test rows" in story.headline
    outcome = next(chapter for chapter in story.chapters if chapter.key == "result")
    assert "caught **none** of the real anomalies" in outcome.text
    # Sem nada marcado não há precisão: o visual compara o que marcou com o que havia.
    assert outcome.bars is not None
    assert [bar.tone for bar in outcome.bars.bars] == ["test", "baseline"]
    assert outcome.bars.bars[0].value == 0
    assert outcome.bars.bars[1].value > 0
    last = story.chapters[-1]
    assert [notice.code for notice in last.notices] == ["nothing_flagged"]


def test_a_run_from_the_history_is_told_without_the_model(tmp_path: Path) -> None:
    before = Lab.create(settings(tmp_path))
    result = run(before)

    after = Lab.create(settings(tmp_path))  # reabrir a app: o run só está no registo
    story = after.history.story(result.run_id)

    assert story.headline == training_story(result).headline
    # Sem a linhagem em memória, a preparação conta-se pelo que a configuração pediu.
    assert "Numbers standardised: mean 0 and standard deviation 1, from the training rows." in (
        story.chapters[0].points
    )


def test_a_reinforcement_run_has_no_story(lab: Lab) -> None:
    job = lab.reinforcement.start(
        ReinforcementConfig.model_validate(
            {
                "agent": {"name": "q_learning", "hyperparameters": {"episodes": 20}},
                "environment": {"name": "gridworld", "parameters": {"layout": "simples"}},
                "seed": 1,
            }
        )
    )
    job.wait(timeout=60)

    with pytest.raises(InvalidParameterError):
        lab.history.story(job.run_id)
