# Camada de dados (`core/data/`)

O único sítio do projeto onde o pandas é usado. Os dados entram como ficheiro, passam por
nove etapas e saem como um `Dataset` com arrays NumPy e metadados. Chega na fase 1.

| # | Etapa | Pasta | Altera os dados? |
|---|---|---|---|
| 1 | Entrada | `io/` | Não |
| 2 | Validação | `validation/` | Não |
| 3 | Perfil | `profiling/` | Não |
| 4 | Limpeza | `cleaning/` | Sim, com registo |
| 5 | Tarefa | `task/` | Não |
| 6 | Divisão | `splitting/` | Separa |
| 7 | Transformação | `transforms/` | Sim |
| 8 | Projeção | `projection/` | Cópia à parte |
| 9 | Saída | `boundary.py` | Converte para NumPy |

## Regras desta camada

1. **Limpeza antes da divisão, transformação depois.** A limpeza não aprende nada com os
   dados; a transformação aprende parâmetros (a média para imputar, o desvio para escalar) e
   por isso só pode ver o treino. Se visse o teste, seria *data leakage*.
2. **Nunca alterar o original.** Cada etapa devolve um DataFrame novo; o ficheiro carregado
   fica intacto.
3. **Tudo fica registado (linhagem).** Cada operação aplicada ("removidas 12 linhas
   duplicadas") fica em `lineage.py`, para a interface a mostrar e a experiência poder ser
   reproduzida.
4. **Explicar em vez de esconder.** Cada etapa produz avisos didáticos ("30% de nulos na
   coluna cabine").
5. **Datasets grandes:** o perfil e os gráficos usam uma amostra; o treino usa tudo.
6. **Nomes de colunas preservados até ao fim**, para chegarem à explicabilidade e à
   interface.

O pandas não sai daqui: `boundary.py` é a fronteira, e um teste de arquitetura falha se
algum ficheiro fora desta pasta o importar.
