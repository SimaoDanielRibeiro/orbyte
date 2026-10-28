# Scripts

Utilitários de linha de comandos. Correm-se a partir da raiz com
`uv run python scripts/<nome>.py`.

| Script | O que faz | Fase |
|---|---|---|
| `download_samples.py` | Descarrega os datasets de exemplo para `data/samples/` e confirma o SHA-256 (`--force` volta a descarregar todos) | 1 |
| `export_openapi.py` | Guarda o OpenAPI da API em `ui/web/openapi.json`, de onde o frontend gera os tipos TypeScript (um teste avisa quando fica desatualizado) | 10 |
| `dev.py` | Arranca a API e a interface web juntas (a interface em <http://localhost:5173>); `Ctrl+C` pára as duas. Na primeira vez, instala as dependências da interface | 10 |

Previsto: recriar a base de dados (fase 5).
