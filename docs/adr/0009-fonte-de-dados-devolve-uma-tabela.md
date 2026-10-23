# 0009 — A fonte de dados devolve uma tabela; o `Dataset` só existe à saída

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

O plano tinha duas versões do contrato `DataSource`: a secção "Adicionar uma fonte de dados"
dizia `DataSource.load() -> Dataset` (numa pasta `data/sources/`), e a tabela de contratos
dizia que a fonte devolve "DataFrame + metadados" (na pasta `data/io/` da árvore). As notas da
fase 0 deixaram a decisão para a fase 1.

Um `Dataset` é o resultado de todas as etapas: limpo, dividido, transformado, em NumPy. Uma
fonte de dados só lê.

## Decisão

- `DataSource.load()` devolve um `LoadedTable`: o DataFrame tal como chegou, o nome, a
  descrição, o alvo habitual, os avisos da leitura e o `SourceSpec` (tipo + parâmetros) que
  permite voltar a obter os mesmos dados.
- As fontes vivem em `core/data/io/` e registam-se em `DATA_SOURCES` com o nome do tipo
  (`"file"`, `"sample"`, `"synthetic"`); `open_source(spec)` cria a fonte a partir da referência.
- O `Dataset` só é criado no fim, pela preparação (`core/data/preparation.py`), na etapa 9.

## Consequências

- A secção "Adicionar uma fonte de dados" de `docs/architecture.md` foi corrigida.
- Uma fonte nova (uma URL, uma base de dados) só tem de ler e devolver uma tabela: limpar,
  dividir e transformar é sempre igual, venham os dados de onde vierem.
- O `SourceSpec` é o que fica guardado com a experiência; a chave curta (`SourceSpec.key`) serve
  de identificador do dataset nos serviços.
