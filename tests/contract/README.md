# Testes de contrato

Testes parametrizados que percorrem os registos e aplicam as mesmas verificações a **todos**
os componentes registados. Um componente novo fica testado sem escrever testes novos.

| Ficheiro | Componentes | Fase |
|---|---|---|
| `test_data_components.py` | Datasets de exemplo, geradores sintéticos, operações de limpeza, estratégias de divisão | 1 |

A partir da fase 2: cada modelo treina em cada tipo de tarefa que declara, `predict` devolve a
forma certa, a mesma seed dá o mesmo resultado, a ficha didática existe; cada métrica devolve
um valor finito.
