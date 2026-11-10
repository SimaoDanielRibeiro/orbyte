# 0058 — O Ollama em 127.0.0.1, e não em localhost

- **Estado:** Proposta
- **Data:** 2026-11-09
- **Muda:** a [0042](0042-agentes-de-ia-com-o-ollama.md) (o endereço por omissão do Ollama)

## Contexto

O endereço por omissão do Ollama era `http://localhost:11434` (`DEFAULT_OLLAMA_URL`, em
`config.py`). Ao verificar o painel do analista (ADR 0057) num Windows, o `GET /llm/status`
demorava sempre 6,1 s, e o painel ficava esse tempo com a caixa da pergunta desligada e o ponto do
modelo a cinzento, de cada vez que se abria a página.

A causa não estava no Ollama nem no projeto, mas no nome `localhost`. Mediu-se um pedido ao Ollama
com a biblioteca padrão do Python (`urllib`, a que o `OllamaServer` usa):

| Endereço | Tempo de um pedido |
| --- | --- |
| `http://localhost:11434` | 2,05 s |
| `http://127.0.0.1:11434` | 0,00 s |

O Windows resolve `localhost` primeiro para o IPv6 (`::1`) e só depois para o IPv4 (`127.0.0.1`).
O Ollama escuta só no IPv4 (o `OLLAMA_HOST` dele é `127.0.0.1:11434`), por isso a ligação ao `::1`
é recusada, e o Windows leva uns 2 s a dar uma ligação recusada como falhada (volta a tentar antes
de desistir). Só então se tenta o `127.0.0.1`. O `status` faz três pedidos ao Ollama (a versão, os
modelos e as capacidades do modelo): 6 s. E não era só o estado: cada vez que o agente de IA
propõe um candidato ou o analista responde, o pedido ao modelo também pagava os 2 s. No macOS e no
Linux, a ligação recusada falha logo, e a diferença quase não se nota.

## Decisão

**O endereço por omissão passa a `http://127.0.0.1:11434`**: o mesmo onde o Ollama escuta de
origem. Quem tiver o Ollama noutro sítio continua a mudá-lo com `ML_LAB_OLLAMA_URL`, e o
`.env.example` diz porque se usa o IP e não o nome.

A razão fica escrita junto do valor (na docstring de `DEFAULT_OLLAMA_URL` e no teste da
configuração): `localhost` e `127.0.0.1` parecem a mesma coisa, e sem o porquê alguém voltaria a
"simplificar".

## Consequências

- Num Windows, o painel do analista fica pronto logo ao abrir, e cada pedido ao modelo deixa de
  perder 2 s.
- O endereço é só IPv4. Um Ollama que escute só no IPv6 (o `OLLAMA_HOST` mudado para `[::1]`)
  precisa do `ML_LAB_OLLAMA_URL`; é um caso raro, e o `GET /llm/status` diz que o Ollama não
  responde, com o endereço que tentou.
- O README (a tabela das variáveis) e o `.env.example` dizem o endereço novo.
