/**
 * Uma escolha de várias opções em fichas (as colunas a remover, as colunas a dividir em
 * intervalos). Uma ficha escolhida ganha um visto, para não depender só da cor.
 */
import { cn } from "cn";
import { Check } from "lucide-react";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

export interface ChipOption {
  value: string;
  label: string;
}

interface ChipGroupProps {
  options: readonly ChipOption[];
  value: readonly string[];
  onChange: (value: string[]) => void;
  /** O nome do grupo, para os leitores de ecrã. */
  label: string;
  id?: string;
  className?: string;
}

export function ChipGroup({ options, value, onChange, label, id, className }: ChipGroupProps) {
  if (options.length === 0) {
    return <p className="text-sm text-muted-foreground">Não há colunas para escolher.</p>;
  }
  return (
    <ToggleGroup
      id={id}
      type="multiple"
      variant="outline"
      size="sm"
      spacing={1.5}
      value={[...value]}
      onValueChange={(chosen) => {
        // Pela ordem das opções (a das colunas), não pela ordem dos cliques.
        onChange(options.map((option) => option.value).filter((one) => chosen.includes(one)));
      }}
      aria-label={label}
      className={cn("flex-wrap justify-start", className)}
    >
      {options.map((option) => {
        const on = value.includes(option.value);
        return (
          <ToggleGroupItem
            key={option.value}
            value={option.value}
            className="rounded-full px-3 font-normal"
          >
            {on && <Check aria-hidden="true" className="size-3.5" />}
            {option.label}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}
