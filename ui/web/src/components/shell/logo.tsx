import type { ComponentProps } from "react";

/**
 * A marca do ML Lab: uma fronteira de decisão a giz entre dois pontos, um de cada classe.
 * É o mesmo desenho do ícone do separador (`public/favicon.svg`).
 */
export function Logo(props: ComponentProps<"svg">) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" {...props}>
      <rect width="32" height="32" rx="7" fill="var(--slate)" />
      <path
        d="M4 22 C11 22 12 9 17 11 S23 23 28 10"
        fill="none"
        stroke="var(--chalk)"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <circle cx="9" cy="11" r="2.6" fill="var(--data-1)" />
      <circle cx="22" cy="24" r="2.6" fill="var(--data-2)" />
    </svg>
  );
}
