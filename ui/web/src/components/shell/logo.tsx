import { useId, type ComponentProps } from "react";

/**
 * A marca do Orbyte (ADR 0051): uma órbita, e só isso. Uma elipse inclinada, numa cor só, com um
 * satélite pequeno no anel; o anel abre-se à volta dele (uma máscara), como nas marcas desenhadas
 * a traço. Sem o núcleo a cores do desenho anterior, lê-se como uma marca, e não como um ícone.
 *
 * É o mesmo desenho do ícone do separador (`public/favicon.svg`), sem o fundo: a barra lateral já
 * é escura. A cor é a do texto forte, por isso serve em qualquer fundo escuro.
 */
export function Logo(props: ComponentProps<"svg">) {
  // Cada logótipo na página precisa da sua máscara (os ids do SVG são da página toda).
  const mask = `orbita-${useId().replace(/:/g, "")}`;
  return (
    <svg viewBox="2 5 28 22" aria-hidden="true" {...props}>
      <defs>
        <mask id={mask} maskUnits="userSpaceOnUse">
          <rect x="0" y="0" width="32" height="32" fill="white" />
          {/* O satélite está no anel, a −40° (um ponto da elipse, já rodado). */}
          <circle cx="22.92" cy="7.81" r="3.9" fill="black" />
        </mask>
      </defs>
      <ellipse
        cx="16"
        cy="16"
        rx="13"
        ry="6.2"
        transform="rotate(-28 16 16)"
        fill="none"
        stroke="var(--text-strong)"
        strokeWidth="2.3"
        mask={`url(#${mask})`}
      />
      <circle cx="22.92" cy="7.81" r="2.35" fill="var(--text-strong)" />
    </svg>
  );
}
