# 0052 — O Parquet e o Excel antigo entram, com o pyarrow e o calamine; e o código fica com a licença MIT

- **Estado:** Proposta
- **Data:** 2026-11-04
- **Muda:** a [0051](0051-orbita-parar-laboratorio-guardar-e-dados-de-verdade.md) (o Parquet e o
  `.xls` deixam de ficar de fora)

## Contexto

Na ADR 0051, o carregador passou a aceitar o JSON, mas o Parquet e o Excel antigo ficaram de
fora: precisavam de bibliotecas novas, e essa decisão era do utilizador. O pedido seguinte foi
aceitá-los, e criar a licença do repositório.

- **O Parquet** é o formato em colunas das ferramentas de dados (o pandas, o Spark, o DuckDB, os
  datasets do Kaggle e do Hugging Face): vem comprimido por dentro e já traz o tipo de cada
  coluna. O pandas lê-o com o `pyarrow`.
- **O `.xls`** é o Excel de 97 a 2003, que ainda aparece em portais de dados públicos e em
  ficheiros antigos. O plano era lê-lo com o `xlrd`; mas, ao correr os testes, o pandas 3.1
  avisou que esse motor está obsoleto e vai sair, e recomenda o `calamine` (`python-calamine`,
  escrito em Rust). Os testes tratam os avisos como erros, e calá-lo era adiar o problema.
- **Os tipos de um Parquet não são os de um CSV.** Um Parquet gravado pelo pandas pode trazer
  categorias, inteiros com nulos (`Int64`), booleanos com nulos, texto com `<NA>` e datas com
  fuso. O perfil lê-os, mas a limpeza, as transformações e o ONNX foram feitos e testados com os
  tipos de um CSV.
- **Sem licença, o código é "todos os direitos reservados"**: no GitHub, ninguém o pode usar,
  mudar ou distribuir legalmente, nem para aprender com ele.

## Decisão

**O Parquet lê-se com o `pyarrow`** (`.parquet` e `.pq`; com um `.gz` por cima, não, porque já
vem comprimido), e o leitor (`_read_parquet`, em `core/data/io/file_source.py`) deixa a tabela
como a de um CSV:

- as categorias passam aos valores (texto, ou números);
- os inteiros com nulos passam a decimais com `NaN`, como num CSV; sem nulos, ficam inteiros;
- os booleanos com nulos passam a `True`/`False`/`NaN`, e o texto com `<NA>` ao texto de sempre;
- as datas com fuso perdem o fuso, e ficam com a hora local de cada uma;
- as células com uma lista ou um objeto passam a texto JSON, como no leitor de JSON;
- o índice que o pandas lá guardou volta como coluna, se tiver nome (com o aviso
  `parquet_index`); sem nome, era só a ordem das linhas, e sai.

**O `.xls` lê-se com o `calamine`**, e o motor do Excel vai sempre dito: o `calamine` para o
`.xls`, o `openpyxl` de sempre para o `.xlsx` e o `.xlsm`. Sem o dizer, com o `calamine`
instalado, o pandas avisa que o motor do `.xlsx` vai mudar; assim, o que já funcionava não muda.

**O `.xls` dos testes é um ficheiro a sério**, gravado pelo próprio Excel como "Livro do Excel
97-2003" (`tests/unit/core/data/io/files/livro-antigo.xls`): hoje nenhuma biblioteca de Python
grava `.xls` sem mais uma dependência. A pasta fica marcada no `.gitattributes` como binária
(`-text`), para o Git nunca lhe mexer nos bytes.

**A licença é a MIT** (`LICENSE`, declarada no `pyproject.toml` e no `package.json`): curta,
permissiva e a mais usada em projetos pessoais e de aprendizagem. Quem quiser pode usar, mudar e
distribuir o código, também num projeto fechado, desde que leve o aviso de copyright; e o código
vem sem garantias. As alternativas pesadas:

- a **Apache 2.0** diz o mesmo, mais uma licença explícita das patentes: faz sentido numa empresa
  com patentes, e é bastante mais longa;
- a **GPL 3.0** obriga quem distribuir uma versão mudada a abri-la também: protege o código, mas
  afasta quem o queira usar num projeto fechado;
- **nenhuma** deixava o código fechado, mesmo num repositório público.

Os datasets de exemplo não são abrangidos: vêm de outras origens, com os seus termos.

## Consequências

- A instalação fica mais pesada: o `pyarrow` descarrega 28 MB e ocupa 87 MB instalado; o
  `calamine`, menos de 2 MB.
- **Com o `pyarrow` instalado, o pandas guarda todo o texto em memória Arrow** (o tipo `str` do
  pandas 3 passa a usá-lo), em todo o laboratório e não só no Parquet. É mais rápido e gasta
  menos memória; todos os testes, também os de ponta a ponta, passaram com ele.
- Um formato novo com o tipo de cada coluna (o Feather, o ORC) entra pelo mesmo caminho: lê-se e
  passa pelo `_plain_column`.
- Quando o `calamine` for o motor por omissão do `.xlsx` no pandas, pode trocar-se o `openpyxl`
  por ele também na leitura (o `openpyxl` continua a ser preciso para gravar).
- O nome no aviso de copyright é o do dono do repositório: muda-se no `LICENSE`, se for preciso.
