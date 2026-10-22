# 0006 — Seeds derivadas por nome a partir de uma seed raiz

- **Estado:** Proposta
- **Data:** 2026-10-22

## Contexto

O plano exige reprodutibilidade: toda a aleatoriedade passa por uma seed explícita, e cada
experiência guarda a sua seed. Uma experiência tem várias partes aleatórias: a divisão dos
dados, a inicialização do modelo, a validação cruzada, o ruído dos dados sintéticos.

Duas abordagens comuns têm problemas:

- **Uma seed global** (`np.random.seed(42)`) é estado global mutável, que as convenções
  proíbem, e é frágil: acrescentar uma chamada aleatória antes de outra muda todos os
  números que vêm a seguir.
- **A mesma seed para todas as partes** torna-as dependentes umas das outras.

Além disso, a fundação só pode usar a biblioteca padrão, por isso não pode criar geradores
do NumPy.

## Decisão

Cada experiência tem uma **seed raiz**, dada pelo utilizador ou gerada com `secrets` e
guardada com a experiência. Cada componente pede a sua seed pelo nome:

```python
seeds = SeedManager.create(config.seed)
split_seed = seeds.seed_for("split")
fold_seed = seeds.child("cv").seed_for("fold_0")
```

A derivação é um hash BLAKE2 de `[raiz, nome]`, reduzido a um inteiro de 32 bits, estável
entre computadores e versões do Python (ao contrário de `hash()`). Os componentes do núcleo
criam os seus geradores a partir da seed recebida: `numpy.random.default_rng(seed)`,
`random_state=seed` no scikit-learn, `torch.Generator().manual_seed(seed)`.

## Consequências

- A mesma seed raiz reproduz a experiência inteira, e trocar o modelo não muda a divisão dos
  dados.
- Não há estado global, e a fundação continua só com a biblioteca padrão.
- Os nomes dos componentes passam a fazer parte da reprodutibilidade: mudar `"split"` para
  `"splitting"` muda os resultados. Um teste fixa valores de referência para detetar
  mudanças acidentais na derivação.
- Não usa o mecanismo nativo do NumPy (`SeedSequence.spawn`); se for preciso, o núcleo pode
  usá-lo a partir da seed derivada.
