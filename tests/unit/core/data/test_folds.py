"""Folds da validação cruzada e colunas de referência nas tarefas sem alvo."""

import numpy as np
import pytest

from ml_lab.core.data.io.base import SourceSpec, open_source
from ml_lab.core.data.preparation import PreparationConfig, prepare, prepare_folds
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.data.transforms.pipeline_builder import TransformConfig
from ml_lab.core.data.transforms.scaling import ScalingMethod
from ml_lab.core.foundation.errors import InvalidParameterError
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.types import Partition, TaskType

IRIS = SourceSpec("sample", {"name": "iris"})
STRATIFIED = PreparationConfig(
    target="species",
    split=SplitConfig(SplitStrategy.STRATIFIED),
    transforms=TransformConfig(scaling=ScalingMethod.STANDARD),
)


def test_each_training_row_is_held_out_in_exactly_one_fold() -> None:
    table = open_source(IRIS).load()
    whole = prepare(table, STRATIFIED, seeds=SeedManager(7)).dataset

    folds = prepare_folds(table, STRATIFIED, seeds=SeedManager(7), k=5)

    held_out = np.concatenate([fold.test.rows for fold in folds])
    assert len(folds) == 5
    assert sorted(held_out.tolist()) == sorted(whole.train.rows.tolist())
    assert not set(held_out.tolist()) & set(whole.test.rows.tolist())  # o teste fica de fora
    for fold in folds:
        assert not set(fold.train.rows.tolist()) & set(fold.test.rows.tolist())


def test_stratified_folds_keep_the_class_proportions() -> None:
    folds = prepare_folds(open_source(IRIS).load(), STRATIFIED, seeds=SeedManager(7), k=5)

    for fold in folds:
        assert fold.test.y is not None
        assert sorted(np.bincount(fold.test.y.astype(np.int64)).tolist()) == [8, 8, 8]


def test_the_scaling_is_learned_again_inside_each_fold() -> None:
    folds = prepare_folds(open_source(IRIS).load(), STRATIFIED, seeds=SeedManager(7), k=5)

    for fold in folds:  # padronizado com a média e o desvio do treino DESTE fold
        assert np.allclose(fold.train.X.mean(axis=0), 0.0, atol=1e-12)
        assert not np.allclose(fold.test.X.mean(axis=0), 0.0, atol=1e-3)


def test_the_number_of_folds_must_make_sense() -> None:
    table = open_source(IRIS).load()

    with pytest.raises(InvalidParameterError, match="folds"):
        prepare_folds(table, STRATIFIED, seeds=SeedManager(7), k=1)


def test_a_large_training_set_is_sampled_keeping_the_class_proportions() -> None:
    table = open_source(IRIS).load()
    whole = prepare(table, STRATIFIED, seeds=SeedManager(7)).dataset  # 120 linhas de treino

    folds = prepare_folds(table, STRATIFIED, seeds=SeedManager(7), k=3, max_rows=60)

    used = np.concatenate([fold.test.rows for fold in folds])
    labels = np.concatenate([fold.test.y for fold in folds if fold.test.y is not None])
    assert len(used) == 60
    assert set(used.tolist()) <= set(whole.train.rows.tolist())  # o teste continua de fora
    assert np.bincount(labels.astype(np.int64)).tolist() == [20, 20, 20]
    step = next(step for step in folds[0].lineage.steps if step.operation == "sample_train")
    assert step.description.startswith("Amostra de 60 linhas das 120 do treino")


def test_the_same_seed_draws_the_same_sample() -> None:
    table = open_source(IRIS).load()

    def sample(seed: int) -> list[int]:
        folds = prepare_folds(table, STRATIFIED, seeds=SeedManager(seed), k=3, max_rows=60)
        return sorted(int(row) for fold in folds for row in fold.test.rows)

    assert sample(7) == sample(7)
    assert sample(7) != sample(8)


def test_a_training_set_that_fits_is_not_sampled() -> None:
    folds = prepare_folds(
        open_source(IRIS).load(), STRATIFIED, seeds=SeedManager(7), k=3, max_rows=500
    )

    assert sum(len(fold.test) for fold in folds) == 120
    assert all(step.operation != "sample_train" for step in folds[0].lineage.steps)


def test_the_sample_needs_a_row_per_fold() -> None:
    with pytest.raises(InvalidParameterError, match="uma linha por fold"):
        prepare_folds(open_source(IRIS).load(), STRATIFIED, seeds=SeedManager(7), k=5, max_rows=3)


def test_a_target_with_an_unsupervised_task_becomes_a_reference() -> None:
    config = PreparationConfig(target="species", task_type=TaskType.CLUSTERING)

    dataset = prepare(open_source(IRIS).load(), config, seeds=SeedManager(7)).dataset

    assert dataset.task_type is TaskType.CLUSTERING
    assert dataset.class_labels == ("setosa", "versicolor", "virginica")
    assert dataset.train.y is not None
    assert "species" not in dataset.feature_names
    task_step = next(step for step in dataset.lineage.steps if step.operation == "choose_task")
    assert task_step.description.startswith("Referência species (não entra no treino")
    assert dataset.splits[Partition.TEST].y is not None
