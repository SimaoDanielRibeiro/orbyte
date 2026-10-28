/**
 * Um controlo deslizante sobre uma lista de valores: os que o backend oferece (os graus de um
 * polinómio, as taxas de aprendizagem, os alphas de uma regularização), por ordem.
 *
 * O controlo anda de valor em valor, e não em passos iguais: numa lista que cresce em ordens de
 * grandeza (0,001 a 2,5), cada valor fica com o mesmo espaço. Os leitores de ecrã ouvem o nome
 * do controlo e o valor escrito, não a posição.
 */
import { cn } from "cn";
import { useId } from "react";

import { Slider } from "@/components/ui/slider";
import { formatValue } from "@/lib/format";

interface OptionSliderProps {
  label: string;
  values: readonly number[];
  value: number;
  onChange: (value: number) => void;
  /** Como escrever cada valor (por omissão, como um hiperparâmetro). */
  format?: (value: number) => string;
  className?: string;
}

export function OptionSlider({
  label,
  values,
  value,
  onChange,
  format = formatValue,
  className,
}: OptionSliderProps) {
  const id = useId();
  const position = nearest(values, value);
  const first = values[0];
  const last = values.at(-1);
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-baseline justify-between gap-3">
        <span id={id} className="text-sm font-medium">
          {label}
        </span>
        <span className="text-sm tabular-nums">{format(values[position] ?? value)}</span>
      </div>
      <Slider
        aria-labelledby={id}
        valueText={format(values[position] ?? value)}
        min={0}
        max={Math.max(0, values.length - 1)}
        step={1}
        value={[position]}
        onValueChange={([index]) => {
          const chosen = values[index ?? position];
          if (chosen !== undefined && chosen !== values[position]) {
            onChange(chosen);
          }
        }}
      />
      {first !== undefined && last !== undefined && (
        <div className="flex justify-between text-xs text-muted-foreground tabular-nums">
          <span>{format(first)}</span>
          <span>{format(last)}</span>
        </div>
      )}
    </div>
  );
}

/** A posição do valor de `values` mais perto de `value`. */
function nearest(values: readonly number[], value: number): number {
  let best = 0;
  values.forEach((option, index) => {
    if (Math.abs(option - value) < Math.abs((values[best] ?? option) - value)) {
      best = index;
    }
  });
  return best;
}
