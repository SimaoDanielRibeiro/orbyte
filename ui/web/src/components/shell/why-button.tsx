/**
 * O botão "Porquê?": abre, ao lado, as ideias por trás do ecrã (os temas didáticos que o
 * backend escreve uma só vez, regra 5 do `CONTRIBUTING.md`).
 */
import { GraduationCap } from "lucide-react";

import { useTopics } from "@/api/topics";
import { Prose } from "@/components/markdown/prose";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

export function WhyButton({ topics }: { topics: readonly string[] }) {
  const explanations = useTopics();
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm">
          <GraduationCap aria-hidden="true" />
          Porquê?
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-xl">
        <SheetHeader>
          <SheetTitle className="font-serif text-2xl font-medium">Porquê?</SheetTitle>
          <SheetDescription>As ideias por trás deste ecrã.</SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-8 px-4 pb-8">
          {topics.map((topic) => {
            const explanation = explanations[topic];
            return explanation === undefined ? null : (
              <section key={topic}>
                <h3 className="text-xl">{explanation.title}</h3>
                <Prose text={explanation.text} className="mt-2" />
              </section>
            );
          })}
        </div>
      </SheetContent>
    </Sheet>
  );
}
