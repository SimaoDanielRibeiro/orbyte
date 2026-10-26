import { useEffect, useState } from "react";

/**
 * O valor, mas só depois de ficar `delay` milissegundos sem mudar: um controlo arrastado faz
 * um pedido no fim, não um por cada posição.
 */
export function useDebounced<T>(value: T, delay: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSettled(value);
    }, delay);
    return () => {
      window.clearTimeout(timer);
    };
  }, [value, delay]);
  return settled;
}
