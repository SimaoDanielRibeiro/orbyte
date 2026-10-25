# Testes de contrato

Testes parametrizados que percorrem os registos e aplicam as mesmas verificações a **todos**
os componentes registados. Um componente novo fica testado sem escrever testes novos.

| Ficheiro | Componentes | O que verifica | Fase |
|---|---|---|---|
| `test_data_components.py` | Datasets de exemplo, geradores sintéticos, operações de limpeza, estratégias de divisão | Dados utilizáveis, tarefa certa, limites dos parâmetros, documentação, partições reproduzíveis | 1 |
| `test_models.py` | Todos os modelos de `MODELS` | Treina em cada tipo de tarefa que declara; `predict` com a forma certa; probabilidades a somar 1; mesma seed, mesmo resultado; modelos iterativos anunciam passos; hiperparâmetros por omissão válidos e descritos; ficha didática com todos os hiperparâmetros; erros de uso explicados | 2 |
| `test_metrics.py` | Todas as métricas de `METRICS` e os avaliadores | Valor finito em dados de exemplo; funções puras; previsões perfeitas nunca pior; explicação escrita; um avaliador para cada tarefa com modelos | 2 |
| `test_search_spaces.py` | O espaço de procura (`x-search`) de todos os modelos de `MODELS` | Lê-se sem erros; cada valor que os agentes podem sortear passa a validação do modelo; cada modelo supervisionado diz o que procurar, ou porque não precisa | 9 |
| `test_vocabulary.py` | Todos os enums que os serviços devolvem | Cada um está no vocabulário, com um nome (único) para cada valor; cada tarefa diz a métrica principal e o modelo sugerido | 10 |
| `test_service_boundary.py` | Todos os métodos públicos dos serviços | Devolvem tipos que o Pydantic descreve; resultados reais passam a JSON estrito e voltam a validar contra o próprio tipo | 4, 10 |
