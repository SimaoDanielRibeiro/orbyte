/**
 * Os dois caminhos desenhados com a sua forma: o manual é uma sequência de passos; o dos
 * agentes é um fluxograma, com o coordenador a distribuir o trabalho pelas famílias.
 */

const MANUAL_STEPS = ["Dados", "Preparar", "Modelo", "Treinar"] as const;

export function ManualDiagram() {
  return (
    <ol
      className="flex flex-wrap items-center gap-y-2 text-sm"
      aria-label="Os passos do teste à mão"
    >
      {MANUAL_STEPS.map((step, index) => (
        <li key={step} className="flex items-center">
          <span className="rounded-md border border-input bg-card px-2.5 py-1">
            <span className="mr-1.5 text-muted-foreground tabular-nums">{index + 1}</span>
            {step}
          </span>
          {index < MANUAL_STEPS.length - 1 && (
            <span aria-hidden="true" className="mx-1.5 h-px w-4 bg-input" />
          )}
        </li>
      ))}
    </ol>
  );
}

export function AgentsDiagram() {
  return (
    <svg
      viewBox="0 0 360 96"
      role="img"
      aria-label="O coordenador distribui o trabalho pelos agentes de cada família de modelos, e o melhor candidato é o vencedor."
      className="h-auto w-full max-w-[360px]"
    >
      <path
        d="M108 48 L136 16 M108 48 L136 48 M108 48 L136 80 M224 16 L252 48 M224 48 L252 48 M224 80 L252 48"
        fill="none"
        stroke="var(--input)"
      />
      <g fill="var(--card)" stroke="var(--input)">
        <rect x="0.5" y="34.5" width="107" height="27" rx="6" />
        <rect x="136.5" y="4.5" width="87" height="23" rx="6" />
        <rect x="136.5" y="36.5" width="87" height="23" rx="6" />
        <rect x="136.5" y="68.5" width="87" height="23" rx="6" />
        <rect x="252.5" y="34.5" width="107" height="27" rx="6" />
      </g>
      <g fill="var(--foreground)" fontSize="13" textAnchor="middle" className="font-sans">
        <text x="54" y="52.5">
          Coordenador
        </text>
        <text x="180" y="20.5">
          Lineares
        </text>
        <text x="180" y="52.5">
          Árvores
        </text>
        <text x="180" y="84.5">
          Vizinhos
        </text>
        <text x="306" y="52.5">
          Vencedor
        </text>
      </g>
    </svg>
  );
}
