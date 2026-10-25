"""Agentes de AutoML: o modo automático, que procura o melhor modelo para um pedido (fase 9).

O pedido é um dataset, um alvo e a métrica a otimizar. Três tipos de agentes trabalham nele,
pela ordem em que um analista o faria à mão (ADR 0025):

1. o **agente de dados** (``data_agent.py``) prepara a tabela a partir do perfil: limpa,
   confirma a tarefa, divide e escolhe as transformações, sempre com a razão;
2. um **agente de modelo** por família (``model_agent.py``): cada um conhece os modelos da
   sua família e o espaço de procura de cada um (``search_space.py``, lido do ``x-search``
   dos hiperparâmetros) e propõe candidatos;
3. o **coordenador** (``coordinator.py``) gere o orçamento: uma triagem com os valores por
   omissão de todos os modelos e uma afinação dos finalistas, por procura aleatória. Mede
   cada candidato com validação cruzada nos folds do treino e nunca vê o teste.

Tudo o que os agentes decidem fica na conversa (``search_log.py``), em português, para se
perceber porque é que o vencedor ganhou. São regras e medições, não um modelo de
linguagem: com a mesma seed, a procura repete-se tal e qual.

O vencedor não sai daqui treinado: o serviço (``services/automl.py``) treina-o como um run
normal, que usa o teste uma única vez e fica no histórico, ao lado dos runs do modo manual.
"""
