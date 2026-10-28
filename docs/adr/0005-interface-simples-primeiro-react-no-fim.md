# 0005 — Interface simples (Streamlit) primeiro, React no fim

- **Estado:** Substituída por 0031 (a interface Streamlit saiu depois da fase 10)
- **Data:** 2026-10-22

## Contexto

Cada funcionalidade do backend deve ser vista e validada cedo. Uma interface em React exige
uma API estável, um sistema de design e muito trabalho de front-end antes de se ver alguma
coisa; deixar toda a interface para o fim adiaria a validação do produto.

## Decisão

A interface existe em duas versões com as mesmas funcionalidades:

- **Simples**, em Streamlit + Plotly, que cresce a par do backend em cada fase e chama os
  serviços diretamente, no mesmo processo;
- **Aprimorada**, em React + TypeScript (fase 9), que muda a apresentação mas não o que se
  pode fazer, e fala com o backend só através da API.

## Consequências

- Tudo é validado cedo, e o redesenho final não obriga a repensar o produto.
- Para as duas versões partilharem tudo: nenhuma lógica vive na interface; os gráficos vêm
  de `visualization_data/`; os formulários vêm das fichas técnicas; os textos didáticos vêm
  do backend.
- O trabalho de interface é feito duas vezes; a versão simples é o rascunho funcional da
  aprimorada.
- O Streamlit volta a correr o script a cada interação: o estado vive em
  `st.session_state` e os recursos pesados em cache (`st.cache_resource`).
