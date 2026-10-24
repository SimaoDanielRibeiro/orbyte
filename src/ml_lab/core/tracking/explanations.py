"""Textos do painel "Porquê?" do ecrã Experiências: registar, comparar e reproduzir."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ml_lab.core.data.explanations import Explanation

TRACKING_EXPLANATIONS: Final[Mapping[str, Explanation]] = MappingProxyType(
    {
        "experiment_tracking": Explanation(
            "Registo de experiências",
            "Cada treino que termina fica registado sozinho: a configuração, a seed, as "
            "métricas, a curva da loss e o que o modelo aprendeu. Chega para comparar runs e "
            "para os repetir. **Guardar** um run junta-lhe o modelo treinado e o treino passo a "
            "passo (a animação e, numa rede neuronal, as camadas em cada snapshot), que pesam "
            "mais: um run guardado volta a abrir com tudo, mesmo depois de fechar a app.",
        ),
        "comparing_runs": Explanation(
            "Comparar runs",
            "Uma comparação só ensina alguma coisa se os runs diferirem numa escolha de cada "
            "vez: o mesmo dataset, a mesma preparação e a mesma divisão, e só um "
            "hiperparâmetro diferente. Se mudarem duas coisas ao mesmo tempo, não se sabe qual "
            "delas fez a diferença. Por isso, a comparação mostra só as escolhas que mudam.",
        ),
        "reproducibility": Explanation(
            "Reprodutibilidade",
            "Um resultado só conta se se puder repetir. No ML Lab, a seed raiz decide todas as "
            "outras (a da divisão, a do modelo, a dos folds), por isso a mesma configuração com "
            "a mesma seed dá exatamente o mesmo modelo e as mesmas métricas. **Reproduzir** um "
            "run treina-o outra vez assim; se as métricas mudarem, alguma coisa não é "
            "determinística, ou os dados mudaram.",
        ),
    }
)
"""Os temas do registo de experiências."""
