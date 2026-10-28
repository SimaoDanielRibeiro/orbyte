/**
 * O ecrã Laboratório: quatro demonstrações de conceitos, cada uma com um controlo e o efeito
 * logo a seguir (o overfitting, a taxa de aprendizagem, o k do KNN e a regularização).
 *
 * A demonstração aberta fica no endereço (`?demo=knn`): a página de um modelo liga-se à que o
 * explica. O que cada demonstração mostra, os valores do seu controlo e a leitura do resultado
 * vêm do backend.
 */
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearch } from "@tanstack/react-router";
import type { ReactNode } from "react";

import type { DemoCard, LaboratoryDemo } from "@/api/generated";
import { laboratoryDemosOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { ErrorNote } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { KnnDemo, LearningRateDemo, OverfittingDemo, RegularizationDemo } from "./demo-panels";

const PANELS: Readonly<Record<LaboratoryDemo, (props: { card: DemoCard }) => ReactNode>> = {
  overfitting: OverfittingDemo,
  learning_rate: LearningRateDemo,
  knn: KnnDemo,
  regularization: RegularizationDemo,
};

export function LaboratoryScreen() {
  const label = useLabel();
  const navigate = useNavigate();
  const search = useSearch({ from: "/laboratorio" });
  const demos = useQuery({ ...laboratoryDemosOptions(), staleTime: "static" });
  const cards = demos.data ?? [];
  const open = cards.find((card) => card.name === search.demo)?.name ?? cards[0]?.name;

  return (
    <Page>
      <PageHeader title={SCREENS.laboratory.title} description={SCREENS.laboratory.description}>
        <WhyButton topics={cards.map((card) => card.topic)} />
      </PageHeader>
      <ErrorNote error={demos.error} className="mt-8" />
      {open === undefined ? (
        demos.error === null && <Skeleton className="mt-10 h-96 w-full" />
      ) : (
        <Tabs
          value={open}
          onValueChange={(demo) => {
            void navigate({ to: "/laboratorio", search: { demo }, replace: true });
          }}
          className="mt-10"
        >
          {/* No telemóvel os separadores partem-se em linhas: a altura acompanha-as. */}
          <TabsList
            variant="line"
            aria-label="As demonstrações"
            className="flex-wrap justify-start group-data-horizontal/tabs:h-auto"
          >
            {cards.map((card) => (
              <TabsTrigger key={card.name} value={card.name}>
                {label("LaboratoryDemo", card.name)}
              </TabsTrigger>
            ))}
          </TabsList>
          {cards.map((card) => {
            const Panel = PANELS[card.name];
            return (
              <TabsContent key={card.name} value={card.name} className="mt-8">
                <Panel card={card} />
              </TabsContent>
            );
          })}
        </Tabs>
      )}
    </Page>
  );
}
