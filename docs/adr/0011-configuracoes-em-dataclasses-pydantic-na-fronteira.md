# 0011 — Configurações da preparação em dataclasses; o Pydantic valida-as na fronteira

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

O plano pede "configuração declarativa e validada": hiperparâmetros e experiências descritos
por esquemas Pydantic, validados à entrada, serializáveis para JSON e reutilizáveis pela
interface para gerar formulários. Mas o Pydantic só entra na fase 2, e a camada de dados da
fase 1 já precisava de descrever a preparação (alvo, limpeza, divisão, transformações) de
forma declarativa e reproduzível.

Havia três caminhos na fase 2: reescrever as configurações da fase 1 em Pydantic (a camada de
dados passava a depender dele), criar modelos Pydantic paralelos e converter entre os dois
(duas definições de cada coisa), ou deixar o Pydantic validar as dataclasses tal como estão.

## Decisão

- As configurações da camada de dados (`SourceSpec`, `PreparationConfig`, `CleaningStep`,
  `SplitConfig`, `TransformConfig`, `ProjectionConfig`) são **dataclasses imutáveis** da
  biblioteca padrão, que se validam no `__post_init__` com `InvalidParameterError` (um
  `ValueError`) e aceitam os valores em texto que chegam de JSON.
- A configuração de uma experiência, `ExperimentConfig` (em `core/schemas/experiment.py`), é
  um modelo **Pydantic** que **compõe** essas dataclasses. O Pydantic valida cada campo pelas
  anotações de tipo, corre o `__post_init__` (os nossos erros chegam como `ValidationError`,
  com a mensagem em português) e serializa tudo para JSON e de volta, sem perdas.
- Os **hiperparâmetros** de cada modelo são subclasses de `Hyperparameters` (Pydantic,
  imutáveis, sem campos desconhecidos), com título e explicação em cada `Field`. O JSON Schema
  que geram é o que as interfaces usam para desenhar o formulário, com duas extensões:
  `"x-scale": "log"` e `"x-labels"`.
- `core/schemas/validation.py` traduz cada tipo de erro do Pydantic numa frase em português
  ("C tem de ser pelo menos 0,0001").

## Consequências

- Cada configuração tem uma só definição, e a camada de dados continua sem depender do
  Pydantic: os seus testes e notebooks usam as dataclasses diretamente.
- O mesmo `ExperimentConfig` serve a interface simples, a API (fase 7) e o registo de
  experiências (fase 5); a ida e volta para JSON está testada.
- As regras das dataclasses escrevem-se à mão no `__post_init__`, em vez de restrições
  declarativas (`Field(gt=0)`); por isso o JSON Schema da preparação é mais pobre. Não faz
  falta: o ecrã Preparar tem controlos próprios, e os formulários gerados são só os dos
  hiperparâmetros e dos geradores sintéticos.
- `core/schemas/` importa `core/data/`; o contrário nunca pode acontecer, para não criar
  ciclos.
