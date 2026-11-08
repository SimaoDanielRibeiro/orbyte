# Decisões de arquitetura (ADR)

Cada decisão importante fica num ficheiro curto, com contexto, decisão e consequências.
Daqui a meses vais saber porque é que algo foi feito assim.

| N.º | Decisão | Estado |
|---|---|---|
| [0001](0001-pytorch-em-vez-de-tensorflow.md) | PyTorch em vez de TensorFlow | Aceite |
| [0002](0002-pandas-isolado-em-core-data.md) | pandas isolado em `core/data/` | Aceite |
| [0003](0003-stack-enxuta-extensoes-atras-de-contratos.md) | Stack enxuta, extensões atrás de contratos | Aceite |
| [0004](0004-sqlite-com-sqlite3.md) | SQLite com `sqlite3` | Aceite |
| [0005](0005-interface-simples-primeiro-react-no-fim.md) | Interface simples (Streamlit) primeiro, React no fim | Substituída por 0031 |
| [0006](0006-seeds-derivadas-por-nome.md) | Seeds derivadas por nome a partir de uma seed raiz | Proposta |
| [0007](0007-descoberta-automatica-de-componentes.md) | Descoberta automática dos componentes registados | Proposta |
| [0008](0008-camada-de-dados-so-com-pandas-e-numpy.md) | Camada de dados só com pandas e NumPy | Proposta |
| [0009](0009-fonte-de-dados-devolve-uma-tabela.md) | A fonte de dados devolve uma tabela; o `Dataset` só existe à saída | Proposta |
| [0010](0010-datasets-de-exemplo-versionados.md) | Datasets de exemplo versionados, com descarga verificada | Proposta |
| [0011](0011-configuracoes-em-dataclasses-pydantic-na-fronteira.md) | Configurações da preparação em dataclasses; o Pydantic valida-as na fronteira | Proposta |
| [0012](0012-contrato-labmodel-e-avaliacao.md) | Contrato LabModel completo desde a fase 2, e avaliação por registos | Proposta |
| [0013](0013-sonda-de-previsoes-e-snapshots-amostrados.md) | Sonda de previsões e snapshots amostrados | Proposta |
| [0014](0014-treino-em-segundo-plano-com-fios-e-controlo.md) | Treino em segundo plano, com um fio de execução e um controlo | Proposta |
| [0015](0015-tarefas-sem-alvo-e-coluna-de-referencia.md) | Tarefas sem alvo no mesmo contrato, com uma coluna de referência para comparar | Proposta |
| [0016](0016-validacao-cruzada-com-a-preparacao-em-cada-fold.md) | Validação cruzada com as transformações aprendidas de novo em cada fold | Proposta |
| [0017](0017-modelos-do-scikit-learn-que-crescem.md) | Modelos do scikit-learn que se mostram a crescer, com o mesmo resultado final | Proposta |
| [0018](0018-explicacoes-independentes-do-modelo.md) | Explicações independentes do modelo, escritas em NumPy | Proposta |
| [0019](0019-fronteira-dos-servicos-pronta-para-a-api.md) | A fronteira dos serviços pronta para a API: dados simples, JSON verificado | Proposta |
| [0020](0020-registar-sempre-guardar-a-pedido.md) | Registo de experiências: registar sempre, guardar a pedido | Aceite |
| [0021](0021-redes-pytorch-com-ciclo-de-treino-observavel.md) | Redes neuronais em PyTorch: um ciclo de treino próprio, que se pode observar | Aceite |
| [0022](0022-api-fina-por-cima-dos-servicos.md) | A API: rotas finas por cima dos serviços, e o treino acompanhado pelo mesmo pedido | Proposta |
| [0023](0023-o-processo-pede-para-nao-ser-travado.md) | O processo pede ao Windows para não ser travado | Proposta |
| [0024](0024-reinforcement-learning-com-os-mesmos-contratos.md) | Reinforcement learning: um paradigma novo, com a mesma maquinaria de treino | Proposta |
| [0025](0025-agentes-de-automl-deterministas.md) | Agentes de AutoML deterministas, por cima dos serviços | Proposta; mudada pela 0042, pela 0049 e pela 0051 |
| [0026](0026-contrato-com-o-frontend.md) | O contrato com o frontend: tipos exatos, um vocabulário e o OpenAPI guardado | Proposta |
| [0027](0027-interface-web-com-o-cliente-gerado.md) | A interface web: React, com o cliente gerado do contrato | Proposta |
| [0028](0028-demonstracoes-dos-modelos-e-formulas-das-metricas.md) | A página de cada modelo e a explicação de cada métrica: demonstrações e fórmulas | Proposta |
| [0029](0029-sistema-visual-ardosia-e-giz.md) | O sistema visual: ardósia e giz | Substituída por 0032 |
| [0030](0030-textos-de-leitura-no-backend.md) | Os textos de leitura vivem no backend, e as interfaces pedem-nos pelo nome | Proposta |
| [0031](0031-retirar-a-interface-streamlit.md) | A interface web é a única: a interface Streamlit sai | Proposta |
| [0032](0032-sistema-visual-da-referencia.md) | O sistema visual da referência: preto, cinzentos e os dados a cores | Proposta; mudada pela 0051 |
| [0033](0033-duas-interfaces-e-a-entrada.md) | Duas interfaces e uma entrada: o Manual e os Agentes, cada um com a sua barra | Proposta |
| [0034](0034-pagina-larga-seccoes-abertas-e-agentes-vivos.md) | A página larga, as secções abertas e o fluxograma vivo dos agentes | Proposta |
| [0035](0035-historico-visual-das-trocas.md) | O histórico visual das trocas: cada mensagem diz a quem se dirige | Proposta |
| [0036](0036-barra-pelos-passos-espaco-e-prosseguir.md) | A barra pelos passos, o espaço em vez dos filetes e o botão de prosseguir | Proposta; mudada pela 0044 |
| [0037](0037-cores-por-familia-barra-que-desliza-e-licao-do-rl.md) | Cores fixas por família, a barra que desliza e a lição do reinforcement learning | Proposta; mudada pela 0048, pela 0049 e pela 0050 |
| [0038](0038-contorno-colorido-sem-numeros-e-menos-texto.md) | O contorno colorido, sem números nem brilho, e menos texto à vista | Proposta |
| [0039](0039-pacote-do-modelo.md) | O pacote do modelo: guardar, levar e voltar a carregar um modelo treinado | Aceite; a disposição do ecrã mudada pela 0051 |
| [0040](0040-uma-thread-para-a-mesma-seed-dar-o-mesmo.md) | Uma só thread nos estimadores que somam em paralelo, para a mesma seed dar o mesmo | Proposta |
| [0041](0041-procura-aberta-e-comparacao-por-grupos.md) | A procura dos agentes fica aberta, e a comparação arruma-se por grupos | Proposta |
| [0042](0042-agentes-de-ia-com-o-ollama.md) | Agentes de IA com o Ollama, ao lado dos agentes de regras | Proposta; seguida pela 0053 |
| [0043](0043-orbyte-e-o-site-em-ingles.md) | Orbyte: o nome novo e o site todo em inglês | Proposta |
| [0044](0044-um-botao-de-prosseguir-letra-maior-barra-alinhada.md) | Um só botão de prosseguir, a letra maior e a barra alinhada | Proposta; mudada pela 0048 |
| [0045](0045-identidade-visual-letra-propria-e-barra-mais-estreita.md) | Identidade visual: uma letra própria nos títulos, rótulos sem maiúsculas e a barra a 18rem | Proposta; mudada pela 0048 |
| [0046](0046-gerar-o-cliente-so-no-dev-no-build-e-na-verificacao.md) | O cliente da API gera-se só no `dev`, no `build` e na verificação | Proposta |
| [0047](0047-a-historia-do-treino-no-ecra-explicar.md) | A história do treino no ecrã Explicar: capítulos com texto e um visual | Proposta; mudada pela 0048 |
| [0048](0048-modelos-por-problema-letra-maior-e-movimento.md) | Os modelos arrumados por problema, a letra maior e um movimento com peso | Proposta; mudada pela 0049 |
| [0049](0049-agentes-com-todos-os-modelos-escolhidos-e-por-palavras.md) | Os agentes com todos os modelos, escolhidos à mão, a falar por palavras; os ícones fixos | Proposta; mudada pela 0050 |
| [0050](0050-barra-que-desliza-pagina-de-erro-troca-e-licao-por-agente.md) | A barra que desliza sem reorganizar a página, a página de erro, a troca que leva ao passo certo e uma lição por agente | Proposta |
| [0051](0051-orbita-parar-laboratorio-guardar-e-dados-de-verdade.md) | Um logótipo só com a órbita, o Parar que para tudo, o Laboratório que se explica, o Guardar em passos e os dados de verdade | Proposta; mudada pela 0052 |
| [0052](0052-parquet-excel-antigo-e-licenca-mit.md) | O Parquet e o Excel antigo entram, com o pyarrow e o calamine; e o código fica com a licença MIT | Proposta |
| [0053](0053-o-analista-um-agente-com-ferramentas.md) | O analista: um agente com ferramentas que fazem as contas, e uma resposta verificada | Proposta; mudada pela 0054 |
| [0054](0054-o-analista-numa-conversa-o-quadro-inteiro-e-pronto-para-outros-computadores.md) | O analista numa conversa com o quadro inteiro, apagar no histórico e o projeto pronto para outros computadores | Proposta |
| [0055](0055-dependencias-da-interface-sem-avisos-de-seguranca.md) | As dependências da interface sem avisos de segurança: três overrides, e a ferramenta do shadcn só quando é precisa | Proposta |

## Como escrever uma ADR

1. Copia o modelo abaixo para `NNNN-titulo-curto.md`, com o próximo número livre.
2. Começa com o estado **Proposta**; passa a **Aceite** ou **Rejeitada** depois de decidir.
   Uma ADR aceite não se reescreve: se a decisão mudar, escreve uma nova e marca a antiga
   como **Substituída por NNNN**.
3. Acrescenta-a à tabela acima.

````markdown
# NNNN — Título da decisão

- **Estado:** Proposta
- **Data:** AAAA-MM-DD

## Contexto

Que problema há para resolver e que forças estão em jogo.

## Decisão

O que se decidiu, numa ou duas frases claras.

## Consequências

O que fica mais fácil, o que fica mais difícil e o que passa a ser preciso fazer.
````
