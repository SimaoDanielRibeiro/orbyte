# 0051 — Um logótipo só com a órbita, o Parar que para tudo, o Laboratório que se explica, o Guardar em passos e os dados de verdade

- **Estado:** Proposta
- **Data:** 2026-11-04
- **Muda:** a [0025](0025-agentes-de-automl-deterministas.md) (parar a procura já não treina o
  melhor até aí), a [0032](0032-sistema-visual-da-referencia.md) (no Laboratório, as explicações
  ficam à vista) e a [0039](0039-pacote-do-modelo.md) (a disposição do Guardar e da biblioteca)

## Contexto

O pedido:

- o logótipo só uma órbita, sem o ponto azul no meio, simples e profissional;
- o Stop dos agentes deve parar tudo e voltar ao painel de uma procura nova;
- o ecrã Models abria sempre na tarefa da preparação (a regressão, por exemplo); deve abrir com
  todas as tarefas;
- o Laboratório mais completo e mais explicado;
- o Guardar mais organizado e profissional;
- uma análise a fundo do código, do arranque aos testes, para o pôr no GitHub;
- ver se falta alguma coisa, se a aplicação aceita todos os tipos de datasets, e testá-la com
  dois grandes.

O diagnóstico:

- **O logótipo eram dois desenhos**: um planeta (o ponto azul) dentro de uma órbita. Em tamanho
  pequeno (a barra, o separador do browser), os dois confundiam-se.
- **O Stop parava a meio**: a procura deixava de medir, mas o melhor até aí ainda treinava e ia
  ao teste, ficava no histórico, e o ecrã continuava na procura parada. Quem carrega em Stop quer
  desistir daquela procura.
- **O Laboratório mostrava, mas não ensinava**: o resumo de cada demonstração estava atrás do
  "?", e não havia nada a dizer o que experimentar nem o que levar dali. Faltavam também duas
  ideias que se usam em todos os modelos: o limiar de decisão e a quantidade de dados.
- **O Guardar era uma faixa de opções**, com a descrição de cada parte escondida no "?", e a
  biblioteca uma grelha de cartões soltos.
- **Os dados de verdade** (dois ficheiros gerados com os problemas de sempre: 151 500 linhas de
  churn em CSV com `;`, Windows-1252, vírgula decimal, datas, nulos e duplicados; 40 000 imóveis
  em Excel com duas folhas e preços impossíveis) passaram o percurso todo, dos serviços à
  interface, sem um erro. Mas mostraram cinco problemas:
  1. a floresta aleatória com 100% no treino e 92,6% no teste recebia "no signs of
     overfitting" (a regra só olhava para os dez pontos de diferença);
  2. 484 cidades repetidas em 150 mil linhas eram "free text";
  3. um id de cliente era "free text" porque as linhas repetidas repetiam o id;
  4. 15 preços de 99 999 999 (erros de digitação) deixavam todos os modelos com R² ≈ 0, e nada
     os sugeria para sair;
  5. com mais de 10 000 linhas, o perfil usa uma amostra, e o mínimo e o máximo vinham dela: os
     valores raros, que são os que importam, podiam não aparecer.
- **Os formatos**: CSV, TSV, TXT (também `.gz`) e Excel. Faltava o JSON, o formato de quem tira
  dados de uma API.
- **Para o GitHub**: nenhum workflow de CI, um README parado na fase 10 e com os botões em
  português, e um texto do backend que chegava ao site em português ("Ficheiro carregado").

## Decisão

**O logótipo é só a órbita**: uma elipse inclinada, num traço, com um satélite pequeno em cima
dela, e uma falha no traço à volta do satélite (uma máscara), para se ler como um objeto em
órbita e não como dois desenhos. Uma só cor (o texto forte), sem o azul; o mesmo desenho no
favicon.

**O Stop para tudo**: depois de parar, nada mais se mede, o vencedor não treina nem vai ao teste,
e nada vai para o histórico (`run_id` e `test` ficam `None`; o aviso `stopped_early` di-lo). O
ecrã volta logo ao pedido, para uma procura nova. A Pausa continua a ser a forma de interromper
sem desistir.

**O Models abre com todas as tarefas.** O filtro fica no endereço (`?task=…`): o "Choose the
model" do Preparar leva-o já com a tarefa preparada, porque é para ela que se escolhe; pela barra,
abre com todas.

**O Laboratório explica-se, com um guia vindo do backend** (`DemoCard.guide`, um `DemoGuide`): a
ideia (o conceito, antes de mexer), os passos a experimentar (cada um diz que valor escolher e o
que ver) e o que fica. Ficam à vista, ao contrário do resto da aplicação (ADR 0032): no
Laboratório, a explicação é o conteúdo. A moldura de cada demonstração lê-se pela ordem em que se
aprende: a ideia, "What you see", a leitura do resultado ao vivo ("What it shows now"), o
controlo com "Try this" ao lado, os gráficos (cada um com o seu nome), e no fim "What to take
away". Duas demonstrações novas:

- **o limiar de decisão** (`threshold`): uma regressão logística em dados desequilibrados (1
  positivo em 10). Cada linha do teste é um ponto, à altura da sua classe e à distância da
  probabilidade que o modelo lhe dá, com a linha do limiar a cortá-las; a matriz de confusão e os
  números (encontrados, perdidos, alarmes falsos); a precisão, o recall, o F1 e a exatidão para
  cada limiar, com a exatidão de responder sempre "negativo". A leitura diz o que se troca por o
  quê, e que a exatidão engana aqui (`accuracy_trap`);
- **a quantidade de dados** (`data_size`): uma árvore sem limite e uma regressão logística
  treinadas com 10 a 750 linhas das luas, cada quantidade com cinco amostras diferentes (a faixa
  mostra o desvio), sempre no mesmo teste; e a fronteira da árvore com as linhas que a treinaram.
  O modelo simples ganha com poucas linhas, o flexível passa-lhe à frente com mais, e a regressão
  logística deixa de melhorar (o viés). As amostras alternam as classes, para nenhuma ficar só com
  uma.

**O Guardar faz-se em três passos numerados**: o run e o nome; o que vai no ficheiro, cada parte
numa linha com o que leva à vista e o interruptor; e o próprio ficheiro, parte a parte. Ao lado
dos dois primeiros, um resumo do pacote acompanha as escolhas: o nome do ficheiro, o tamanho, o
que ele permite (prever; correr fora do Orbyte, em ONNX) e os avisos. A biblioteca é uma lista,
uma linha por pacote (a marca da família, o nome, o modelo e os dados, a métrica, o tamanho, a
data, a origem e as ações), com a faixa de carregar um `.mllab` por cima.

**Os dados de verdade, um a um:**

1. **O overfitting conta também o erro** (`memorised`, em `evaluation/diagnostics.py`): numa
   métrica de 0 a 1, é overfitting com dez pontos de diferença, como antes, ou com cinco pontos
   e o que falha no teste (o erro; no R², a variação por explicar) a mais do dobro do do treino,
   desde que a diferença não caiba no ruído do teste (duas vezes √(p(1 − p)/n), com p o que
   falha). Sem esta última condição, o Iris (3 erros em 30 linhas de teste) também era
   overfitting. A mesma regra serve o aviso do treino, a história do treino e os agentes, para
   as três leituras nunca se contradizerem.
2. **Categorias a mais não é texto livre** (`repeated_categories`): um texto com mais de 50
   valores que se repetem, em média, cinco vezes ou mais é "too many categories to encode"; a
   explicação sugere uma coluna mais grossa (a região em vez da cidade). Continua a ficar de fora
   do modelo: 484 colunas de one-hot não ajudavam.
3. **Um identificador pode ter repetidos**: um código sem espaços com 95% de valores diferentes
   é um identificador (antes, tinham de ser 100%).
4. **Os erros grosseiros do alvo sugerem-se para sair**: num alvo numérico, poucos valores (até 1%
   das linhas) para lá de dez IQR dos quartis são quase sempre erros. A sugestão é
   `treat_outliers` com `remove` e o fator 10: tira essas linhas, e não limita os valores
   (limitar o alvo era inventar a resposta). Nos imóveis, o R² passou de ≈ 0 para 0,82 (a
   regressão linear) e 0,87 (o vencedor dos agentes). A regra do IQR habitual continua a só
   avisar.
5. **Os extremos do perfil vêm da coluna inteira**: o mínimo, o máximo e os valores muito
   afastados; as outras estatísticas continuam a vir da amostra.

**O JSON entra**: uma lista de registos, JSON Lines (`.jsonl`, `.ndjson`), os registos dentro de
uma só chave (`{"data": [...]}`) e as colunas (`{"idade": [...]}`), também em `.gz`. Os objetos
aninhados ficam planos (`morada.cidade`), as listas ficam como texto, e um aviso
(`json_shape`) diz como se leu, quando não foi uma lista de registos. Sem dependências novas
(o `json` da biblioteca padrão e o `json_normalize` do pandas).

**Para o GitHub:**

- um workflow (`.github/workflows/ci.yml`): o backend (ruff, mypy, pytest) e a interface (o
  formato, o lint, os tipos, os testes e o build) em cada push e pull request; os testes de ponta
  a ponta, que demoram, no `main` e à mão, com o relatório guardado quando falham;
- o `.gitattributes` com `* text=auto`: os ficheiros ficam com LF no repositório, seja qual for
  o sistema de quem os grava (o CI corre em Linux);
- o README refeito: o nome, as fases até à 13, os nomes dos ecrãs e dos botões como o site os
  mostra, os formatos aceites, as variáveis do Ollama e as verificações certas (`npm run check`);
- o último texto em português que chegava ao site passou a inglês ("Uploaded file: …").

## Consequências

- Parar uma procura deita-a fora: quem quiser o melhor até aí pausa, ou treina-o à mão a partir
  do quadro dos candidatos (`train_candidate`).
- Uma demonstração nova do Laboratório precisa do seu guia (o teste do serviço exige a ideia, pelo
  menos três passos e o que fica), além do resumo, do tema e do controlo.
- A regra do overfitting passa a precisar do número de linhas do teste: quem a chama sem ele (a
  validação cruzada dos agentes, que usa o treino todo) não desconta o ruído.
- O perfil lê a coluna inteira mais uma vez para os extremos: em 150 mil linhas, é coisa de
  milissegundos.
- O Parquet e o Excel antigo (`.xls`) ficaram de fora: precisavam de bibliotecas novas, e isso
  ficou para decidir. Entraram logo a seguir, na [0052](0052-parquet-excel-antigo-e-licenca-mit.md).
- O CI do GitHub Actions instala o PyTorch do PyPI, que no Linux traz as bibliotecas da CUDA:
  a primeira corrida é lenta, as seguintes usam a cache do uv.
