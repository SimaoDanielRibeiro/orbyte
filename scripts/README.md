# Scripts

Utilitários de linha de comandos. Correm-se a partir da raiz com
`uv run python scripts/<nome>.py`.

| Script | O que faz | Fase |
|---|---|---|
| `download_samples.py` | Descarrega os datasets de exemplo para `data/samples/` e confirma o SHA-256 (`--force` volta a descarregar todos) | 1 |

Previsto: recriar a base de dados (fase 5).
