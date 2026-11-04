import * as React from "react";

/**
 * Abaixo disto a barra lateral é uma gaveta: num tablet ao alto (768 px), 18rem de barra fixa
 * mais as margens deixavam ao conteúdo menos espaço do que no telemóvel. É o `lg` do Tailwind, e
 * as classes da barra (`lg:block`, `lg:hidden`) usam o mesmo ponto.
 */
const MOBILE_BREAKPOINT = 1024;
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
 * Se o ecrã é estreito como o de um telemóvel ou de um tablet ao alto (a barra lateral passa a
 * gaveta). O nome vem do shadcn/ui, que o usa na barra.
 *
 * Segue a media query com `useSyncExternalStore`, a forma do React de ler um estado de fora:
 * sem um efeito a copiar o valor para o estado, que obrigava a desenhar duas vezes.
 */
export function useIsMobile(): boolean {
  return React.useSyncExternalStore(subscribe, isMobileNow, () => false);
}
