# 0013 — Sonda de previsões e snapshots amostrados

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

A fase 3 tem de mostrar o treino passo a passo: a fronteira de decisão da regressão
logística a mexer-se, a reta da regressão linear a ajustar-se. Há três problemas.

1. **O que desenhar.** Uma fronteira é onde as previsões do modelo mudam. Para a desenhar é
   preciso perguntar ao modelo o que prevê em muitos pontos, em cada snapshot. Se cada
   modelo calculasse a sua própria fronteira (a partir dos pesos, dos centróides, da
   árvore…), a interface precisaria de código por modelo, e os dados vivem em mais
   dimensões do que o ecrã.
2. **Quantos snapshots.** Guardar o estado a cada passo pode dar milhares de fotografias
   (20 000 épocas × 2500 previsões). Mas não se sabe à partida quantos passos o treino vai
   ter, porque o modelo pode convergir mais cedo.
3. **O último estado.** O resultado final tem de aparecer sempre na animação, mesmo que o
   último passo não calhe no intervalo de amostragem.

## Decisão

- **Sonda.** O contexto de treino tem um atributo `probe`: os pontos (no espaço das
  features) onde quem observa quer as previsões. Em cada snapshot, o modelo junta ao seu
  estado as previsões nesses pontos (`LabModel.snapshot_state(probe)`: as probabilidades,
  se as der; senão, as previsões). A grelha é desenhada no **plano da projeção 2D** do ecrã
  Preparar e levada de volta ao espaço das features com `Projection.to_features` (as
  direções que o plano não mostra ficam na média do treino). Com uma só feature, a grelha é
  uma reta.
- **Amostragem adaptativa.** O `SnapshotSampler` começa por querer todos os passos e,
  sempre que já há mais de 60 snapshots, duplica o intervalo e deita fora os que deixaram
  de calhar nele. O passo 1 fica sempre. O mesmo amostrador é partilhado pelo contexto (que
  só fotografa quando ele quer) e pelo gravador (que desbasta o que já guardou).
- **Fotografia do fim.** O `Trainer` fotografa o último passo se o modelo não o fez. Os
  modelos que treinam de uma vez (os do scikit-learn) ficam com uma única fotografia, a do
  modelo final.
- **Só números no núcleo.** `visualization_data/` transforma os snapshots num
  `TrainingAnimation`: um campo por fotografia (a probabilidade da classe positiva, a classe
  prevista ou o valor previsto), os eixos e a nuvem de pontos de treino. As interfaces só
  desenham.

## Consequências

- Qualquer modelo, didático ou do scikit-learn, mostra a sua fronteira sem código novo:
  basta prever. Os modelos da fase 4 (KNN, árvores, SVM…) e a MLP da fase 6 ficam animados
  de graça.
- A memória fica limitada: no máximo 61 fotografias por run, espalhadas por todo o treino.
- Com mais de duas features, a fronteira desenhada é um corte da verdadeira, pelo meio dos
  dados; a interface explica-o no painel "Porquê?".
- Cada snapshot custa uma previsão em 2500 pontos. Com a amostragem, isso só acontece
  algumas dezenas de vezes por treino.
- As previsões na sonda vão nos eventos como arrays NumPy. A API da fase 7 terá de as
  converter para JSON antes de as enviar pelo WebSocket.
