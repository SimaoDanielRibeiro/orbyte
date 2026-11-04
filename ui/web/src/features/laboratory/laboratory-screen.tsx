/**
 * O ecrã Laboratório: seis demonstrações de conceitos, cada uma com um controlo e o efeito logo
 * a seguir (o overfitting, a taxa de aprendizagem, o k do KNN, a regularização, o limiar de
 * decisão e a quantidade de dados).
 *
 * A demonstração aberta fica no endereço (`?demo=knn`): a página de um modelo liga-se à que o
 * explica. O que cada demonstração mostra, o seu guia (a ideia, o que experimentar e o que fica),
 * os valores do seu controlo e a leitura do resultado vêm do backend (ADR 0051).
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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import {
  DataSizeDemo,
  KnnDemo,
  LearningRateDemo,
  OverfittingDemo,
  RegularizationDemo,
  ThresholdDemo,
} from "./demo-panels";

const PANELS: Readonly<Record<LaboratoryDemo, (props: { card: DemoCard }) => ReactNode>> = {
  overfitting: OverfittingDemo,
  learning_rate: LearningRateDemo,
  knn: KnnDemo,
  regularization: RegularizationDemo,
  threshold: ThresholdDemo,
  data_size: DataSizeDemo,
};

export function LaboratoryScreen() {
  const label = useLabel();
  const navigate = useNavigate();
  const search = useSearch({ from: "/lab" });
  const demos = useQuery({ ...laboratoryDemosOptions(), staleTime: "static" });
  const cards = demos.data ?? [];
  const open = cards.find((card) => card.name === search.demo)?.name ?? cards[0]?.name;

  return (
    <Page>
      <PageHeader
        title={SCREENS.laboratory.title}
        description={SCREENS.laboratory.description}
        topics={cards.map((card) => card.topic)}
      />
      <ErrorNote error={demos.error} className="mt-8" />
      {open === undefined ? (
        demos.error === null && <Skeleton className="mt-10 h-96 w-full" />
      ) : (
        <Tabs
          value={open}
          onValueChange={(demo) => {
            void navigate({ to: "/lab", search: { demo }, replace: true });
          }}
          className="mt-10"
        >
          {/* No telemóvel os separadores partem-se em linhas: a altura acompanha-as. */}
          <TabsList
            variant="line"
            aria-label="The demos"
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
