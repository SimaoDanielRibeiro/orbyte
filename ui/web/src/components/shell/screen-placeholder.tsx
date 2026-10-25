import { Page, PageHeader } from "./page";
import type { Screen } from "./screens";

interface ScreenPlaceholderProps {
  screen: Screen;
  /** A etapa da fase 10 em que o ecrã chega (`10.4`). */
  milestone: string;
}

/** Um ecrã que ainda não foi feito: diz o que vai fazer e em que etapa chega. */
export function ScreenPlaceholder({ screen, milestone }: ScreenPlaceholderProps) {
  return (
    <Page>
      <PageHeader title={screen.title} description={screen.description} />
      <p className="mt-12 max-w-[68ch] border-t pt-6 text-muted-foreground">
        Este ecrã chega na etapa {milestone} da fase 10. Até lá, está na interface simples.
      </p>
    </Page>
  );
}
