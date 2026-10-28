import { Link, useRouter, type ErrorComponentProps } from "@tanstack/react-router";

import { ApiRequestError } from "@/api/errors";
import { Button } from "@/components/ui/button";

import { Logo } from "./logo";
import { Page, PageHeader } from "./page";

/**
 * Quando a página não se consegue abrir. O caso mais comum é a API não estar a correr: a
 * interface precisa dela para tudo, a começar pelo vocabulário.
 */
export function RootError({ error }: ErrorComponentProps) {
  const router = useRouter();
  const unreachable = error instanceof ApiRequestError && error.unreachable;
  const message = error instanceof Error ? error.message : "Aconteceu um erro inesperado.";

  return (
    <div className="flex min-h-svh items-center">
      <Page width="reading">
        <Logo className="mb-8 h-8 w-10" />
        <PageHeader
          title={unreachable ? "A API não responde" : "Esta página não abriu"}
          subtitle={
            unreachable
              ? "A interface pede tudo à API, a começar pelos nomes do vocabulário. Arranca-a num terminal, na raiz do projeto:"
              : message
          }
        />
        {unreachable && (
          <pre className="mt-6 rounded-lg border bg-card px-4 py-3 text-sm">
            <code>uv run python -m ml_lab.api</code>
          </pre>
        )}
        <Button
          className="mt-8"
          onClick={() => {
            void router.invalidate();
          }}
        >
          Tentar outra vez
        </Button>
      </Page>
    </div>
  );
}

/** Um endereço que não é de nenhum ecrã. */
export function NotFound() {
  return (
    <Page width="reading">
      <PageHeader
        title="Este ecrã não existe"
        subtitle="O endereço não corresponde a nenhum ecrã do ML Lab."
      />
      <Button asChild variant="outline" className="mt-8">
        <Link to="/">Ir para o início</Link>
      </Button>
    </Page>
  );
}
