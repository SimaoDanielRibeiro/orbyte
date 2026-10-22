# 0007 — Descoberta automática dos componentes registados

- **Estado:** Proposta
- **Data:** 2026-10-22

## Contexto

Os componentes registam-se com um decorador (`@MODELS.register("knn")`), e um decorador só
corre quando o seu módulo é importado. Se ninguém importar
`models/supervised/classification/extra_trees.py`, o modelo nunca aparece no catálogo.

A solução habitual, uma lista de imports num `__init__.py`, obriga a editar um ficheiro
existente sempre que se acrescenta um algoritmo. Isso contraria o objetivo "adicionar um
algoritmo = criar um ficheiro e registá-lo, sem alterar código existente".

## Decisão

O `Registry` aceita `discover="<pacote>"`. Na primeira consulta (`get`, `names`, `for_task`,
`in`, `len`...), importa recursivamente todos os módulos desse pacote com
`import_submodules`. Por exemplo:

```python
MODELS = Registry[type[LabModel]](
    "models",
    task_types=lambda model: model.info.task_types,
    discover="ml_lab.core.models",
)
```

A descoberta corre uma só vez, está protegida por um lock (as sessões do Streamlit e os
pedidos da API correm em fios diferentes) e não esconde erros de importação.

## Consequências

- Um ficheiro novo na pasta certa fica registado sem mexer em mais nada, e os testes de
  contrato apanham-no sozinhos.
- Importar o pacote não carrega tudo de imediato: o custo (por exemplo, importar o PyTorch
  na fase 6) só se paga quando o catálogo é preciso.
- Todos os módulos do pacote descoberto são importados, mesmo os que não registam nada: os
  módulos não devem ter efeitos ao serem importados, além de registar componentes.
- Um erro de importação num algoritmo faz falhar a consulta ao registo inteiro, de
  propósito: um algoritmo partido não pode desaparecer do catálogo em silêncio.
- Os testes de arquitetura leem os imports do código-fonte, por isso não veem os imports
  feitos pela descoberta; isso não é um problema, porque a descoberta só importa o pacote
  indicado.
