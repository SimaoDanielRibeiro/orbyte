# 0023 — O processo pede ao Windows para não ser travado

- **Estado:** Proposta
- **Data:** 2026-10-24

## Contexto

Ao preparar o teste da fase 7, uma MLP de 150 épocas demorava 9 s, e o perfil não mostrava
nada de errado no código. A investigação foi por eliminação:

- o adaptador do PyTorch era pouco mais lento do que um ciclo de treino escrito à mão, no
  mesmo processo;
- o abrandamento parecia vir de importar o scikit-learn: nos processos onde ele estava, uma
  época passava de 3,5 para 19,5 ms. Mas desligar o garbage collector, os fios do OpenMP e do
  BLAS, o `torch._dynamo` ou os números subnormais não mudava nada;
- o teste decisivo foi esperar sem importar nada: ao fim de cerca de um segundo, o treino
  ficava cinco vezes mais lento na mesma.

A causa é o Windows 11. Um processo sem janela em primeiro plano (o Streamlit, a API ou os
testes, lançados de um terminal) passa a modo de poupança de energia (*EcoQoS*). Num
processador híbrido como o i7-13620H, com núcleos de desempenho e de eficiência, isso
significa os núcleos lentos ou uma frequência baixa. Os imports só pareciam culpados porque
demoravam esse segundo.

## Decisão

`ml_lab.process.run_at_full_speed()` pede ao Windows, só para o próprio processo, que não
lhe aplique a poupança. Usa a API documentada `SetProcessInformation` com
`ProcessPowerThrottling`. Chama-se no arranque da interface simples, da API e dos testes.
Noutros sistemas operativos não faz nada.

## Consequências

- Neste portátil, uma época da MLP passa de 19,5 para 3,5 ms, e uma experiência de 150
  épocas de 9 s para menos de 1 s. O resto do ML Lab (o scikit-learn, a preparação dos
  dados) também deixa de ser travado.
- Não muda nenhuma definição do sistema, e o efeito acaba com o processo.
- Com a app a correr, o portátil gasta mais energia, porque o processo deixa de ser tratado
  como trabalho de fundo.
- Fica uma lição para medir desempenho: comparar sempre no mesmo processo e à mesma
  distância do arranque, porque o sistema operativo também mexe nos resultados.
