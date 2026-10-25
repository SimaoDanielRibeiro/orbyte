import * as React from "react";

const MOBILE_BREAKPOINT = 768;
const MOBILE_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

function subscribe(onChange: () => void): () => void {
  const query = window.matchMedia(MOBILE_QUERY);
  query.addEventListener("change", onChange);
  return () => {
    query.removeEventListener("change", onChange);
  };
}

function isMobileNow(): boolean {
  return window.matchMedia(MOBILE_QUERY).matches;
}

/**
 * Se o ecrã é estreito como o de um telemóvel (a barra lateral passa a gaveta).
 *
 * Segue a media query com `useSyncExternalStore`, a forma do React de ler um estado de fora:
 * sem um efeito a copiar o valor para o estado, que obrigava a desenhar duas vezes.
 */
export function useIsMobile(): boolean {
  return React.useSyncExternalStore(subscribe, isMobileNow, () => false);
}
