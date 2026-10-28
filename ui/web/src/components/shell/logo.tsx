import type { ComponentProps } from "react";

/**
 * A marca do ML Lab: uma fronteira de decisão entre dois pontos, um de cada classe, nas duas
 * primeiras cores dos dados. É o mesmo desenho do ícone do separador (`public/favicon.svg`),
 * sem o fundo: a barra lateral já é escura.
 */
export function Logo(props: ComponentProps<"svg">) {
  return (
    <svg viewBox="2 6 28 21" aria-hidden="true" {...props}>
      <path
        d="M5 21.5 C11 21.5 12.5 10 17 11.5 S22.5 22 27 10.5"
        fill="none"
        stroke="var(--text-strong)"
        strokeWidth="2.6"
        strokeLinecap="round"
      />
      <circle cx="9.5" cy="11" r="2.8" fill="var(--data-1)" />
      <circle cx="21.5" cy="23.5" r="2.8" fill="var(--data-2)" />
    </svg>
  );
}
