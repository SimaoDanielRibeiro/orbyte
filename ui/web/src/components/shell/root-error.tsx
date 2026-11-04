import { Link, useRouter, type ErrorComponentProps } from "@tanstack/react-router";

import { ApiRequestError } from "@/api/errors";
import { Button } from "@/components/ui/button";

import { Logo } from "./logo";
import { Page } from "./page";

/**
 * Quando a página não se consegue abrir (ADR 0050): o desenho de um gráfico em que a linha sobe,
 * como um treino a correr bem, e cai a pique; por baixo, o que aconteceu e "Try again".
 *
 * O caso mais comum é a API não estar a correr: a interface precisa dela para tudo, a começar
 * pelo vocabulário. Aí, o gráfico diz "API offline" e a página mostra o comando que a arranca.
 */
export function RootError({ error, reset }: ErrorComponentProps) {
  const router = useRouter();
  const unreachable = error instanceof ApiRequestError && error.unreachable;
  const message = error instanceof Error ? error.message : "Something unexpected went wrong.";

  return (
    <main className="flex min-h-svh items-center justify-center bg-background px-4 py-12 md:px-8">
      <div className="w-full max-w-xl">
        <Logo className="mb-10 h-8 w-10" />
        <FallingChart label={unreachable ? "API offline" : "error"} />
        <h1 className="mt-10 text-title">
          {unreachable ? "The API isn't responding" : "Something went wrong"}
        </h1>
        <p className="mt-3 max-w-[56ch] text-base text-soft">
          {unreachable
            ? "The interface asks the API for everything, starting with the vocabulary's names. Start it in a terminal, at the project's root, and try again:"
            : message}
        </p>
        {unreachable && (
          <pre className="mt-5 rounded-xl border bg-card px-4 py-3 text-sm">
            <code>uv run python -m ml_lab.api</code>
          </pre>
        )}
        <div className="mt-8 flex flex-wrap gap-3">
          <Button
            size="lg"
            onClick={() => {
              reset();
              void router.invalidate();
            }}
          >
            Try again
          </Button>
          {/* Um endereço a sério (e não uma ligação do router): volta a carregar tudo do zero. */}
          <Button asChild size="lg" variant="outline">
            <a href="/">Back to the home page</a>
          </Button>
        </div>
      </div>
    </main>
  );
}

/** Um endereço que não é de nenhum ecrã: o mesmo desenho, com a linha a cair para o 404. */
export function NotFound() {
  return (
    <Page width="reading">
      <FallingChart label="404" />
      <h1 className="mt-10 text-title">This screen doesn&apos;t exist</h1>
      <p className="mt-3 text-base text-soft">The address matches no Orbyte screen.</p>
      <Button asChild size="lg" variant="outline" className="mt-8">
        <Link to="/">Go to the home page</Link>
      </Button>
    </Page>
  );
}

/**
 * O desenho: uma grelha, uma curva que sobe aos poucos (desenha-se primeiro) e, no ponto mais
 * alto, a queda a vermelho até ao fundo, onde fica um ponto a pulsar e o nome do que caiu. As
 * animações são as classes `error-*` de `globals.css`; com "reduzir movimento", aparece já tudo
 * desenhado. É decoração: o que importa está escrito por baixo.
 */
function FallingChart({ label }: { label: string }) {
  return (
    <svg viewBox="0 0 520 220" aria-hidden="true" className="w-full overflow-visible">
      {[30, 80, 130, 180].map((y) => (
        <line key={y} x1={36} x2={510} y1={y} y2={y} stroke="#1f1f1f" />
      ))}
      <line x1={36} x2={36} y1={20} y2={190} stroke="#2e2e2e" />
      <line x1={36} x2={510} y1={190} y2={190} stroke="#2e2e2e" />
      {/* A subida: o treino a correr bem. */}
      <path
        d="M36 168 C 70 160, 92 164, 120 150 S 172 132, 204 128 S 262 104, 292 101 S 344 74, 372 66"
        pathLength={1}
        fill="none"
        stroke="var(--text-soft)"
        strokeWidth={2.5}
        strokeLinecap="round"
        className="error-rise"
      />
      <circle cx={372} cy={66} r={4} fill="var(--text-strong)" className="error-peak" />
      {/* A queda, a pique, e a linha parada no fundo. */}
      <path
        d="M372 66 L 398 186"
        pathLength={1}
        fill="none"
        stroke="var(--negative)"
        strokeWidth={2.5}
        strokeLinecap="round"
        className="error-fall"
      />
      <path
        d="M398 186 L 510 186"
        fill="none"
        stroke="var(--negative)"
        strokeWidth={2}
        strokeDasharray="3 7"
        className="error-flat"
      />
      <circle cx={398} cy={186} r={5} fill="none" stroke="var(--negative)" className="error-ping" />
      <circle cx={398} cy={186} r={5} fill="var(--negative)" className="error-dot" />
      <text
        x={410}
        y={172}
        className="error-label fill-negative font-mono text-[15px] tracking-wide"
      >
        {label}
      </text>
    </svg>
  );
}
