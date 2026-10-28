/**
 * A seed raiz de um pedido: um inteiro de 0 a 2³² − 1 (o que o `SeedManager` do backend aceita;
 * fora disso, a API responde `invalid_seed`). Os formulários verificam-na antes de pedir.
 */
export const MAX_SEED = 2 ** 32 - 1;

export function isValidSeed(seed: number): boolean {
  return Number.isInteger(seed) && seed >= 0 && seed <= MAX_SEED;
}
