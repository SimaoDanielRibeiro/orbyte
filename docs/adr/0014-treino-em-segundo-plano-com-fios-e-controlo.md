# 0014 — Treino em segundo plano, com um fio de execução e um controlo

- **Estado:** Proposta
- **Data:** 2026-10-23

## Contexto

O ecrã Treinar tem de mostrar o progresso enquanto o modelo treina, e deixar pausar,
retomar e cancelar. Na fase 2, o serviço de experiências treinava no fio da interface: o
ecrã ficava preso até ao fim, sem progresso nem forma de parar. Mais tarde, a API (fase 7)
vai precisar do mesmo: lançar um treino e devolver a resposta antes de ele acabar.

As opções eram processos (`multiprocessing`), `asyncio` ou fios de execução (`threading`).
Os processos isolam melhor, mas obrigariam a copiar os dados e o modelo entre processos, e
os eventos passariam por filas. O `asyncio` obrigaria todos os modelos a cederem o controlo
de propósito. O cálculo pesado é feito pelo NumPy, que liberta o GIL, por isso um fio de
execução não atrasa a interface.

## Decisão

- **`TrainingJob`** corre o trabalho num fio de execução próprio (`daemon`) e guarda o
  resultado ou o erro, que `wait()` devolve ou relança no fio de quem espera. O estado
  (`À espera`, `A treinar`, `Em pausa`, `Terminado`, `Falhou`) e o progresso lêem-se a
  qualquer momento: o progresso vem do gravador de snapshots, que é seguro entre fios.
- **`TrainingControl`** leva os pedidos da interface ao treino, com dois `threading.Event`.
  O contexto de treino espera no fim de cada passo enquanto houver uma pausa, e o modelo
  pergunta-lhe `should_stop()` no fim de cada passo.
- **Cancelar não é um erro.** O treino pára no fim do passo atual, o modelo fica com o que
  aprendeu, é avaliado na mesma e o resultado leva um aviso a dizer em que passo parou.
- **O serviço valida antes de lançar.** `ExperimentService.start` faz logo o plano do run
  (seed, modelo, hiperparâmetros, dados, tarefa), por isso uma configuração inválida falha
  no ecrã Modelo e não dentro do fio. Só o treino e a avaliação correm em segundo plano.
- **Na interface simples**, o ecrã Treinar volta a ler o progresso a cada meio segundo, num
  `st.fragment(run_every=...)`, e redesenha o ecrã inteiro quando o treino acaba. O
  `preview(run_id)` do serviço dá a fronteira do último snapshot enquanto o treino decorre.

## Consequências

- A interface nunca fica presa, e a mesma peça serve a API da fase 7.
- Pausar e parar só acontecem no fim de um passo: um modelo que treina de uma vez (os do
  scikit-learn) não se pode pausar a meio, só esperar que acabe.
- Um fio de execução não se pode matar à força: um modelo iterativo tem de perguntar
  `should_stop()` em cada passo (os testes de contrato confirmam-no).
- Os subscritores do `EventBus` passam a ser chamados no fio do treino; o `EventBus` e o
  gravador protegem-se com `threading.Lock`.
- Um treino lançado e esquecido continua a correr até ao fim. A interface manda parar o
  anterior quando se lança um novo.
