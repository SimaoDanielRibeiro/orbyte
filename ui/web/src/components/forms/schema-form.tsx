/**
 * Um formulário gerado a partir de um JSON Schema: os hiperparâmetros de um modelo, os
 * parâmetros de um gerador de dados, as definições dos agentes.
 *
 * Um controlo por propriedade, sem código próprio para cada uma. Entende as extensões do
 * laboratório:
 *
 * - `x-scale: "log"`: números que variam em ordens de grandeza (o `C` de um SVM) andam em
 *   passos 1, 2, 5 de cada potência de 10. Os inteiros com uma gama muito larga (de 5 a 5000
 *   épocas) também;
 * - `x-labels`: o nome a mostrar de cada opção de um `enum`.
 */
import { cn } from "cn";

import { Explain } from "@/components/explain/explain";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatValue } from "@/lib/format";

import { niceStep, schemaProperties, stepOptions, type Property, type Values } from "./schema";

interface SchemaFormProps {
  schema: Record<string, unknown>;
  values: Values;
  onChange: (values: Values) => void;
  /** O começo dos ids dos controlos (tem de ser único na página). */
  idPrefix: string;
  className?: string;
}

export function SchemaForm({ schema, values, onChange, idPrefix, className }: SchemaFormProps) {
  return (
    <div className={cn("flex flex-col divide-y", className)}>
      {schemaProperties(schema).map(([name, property]) => (
        <Control
          key={name}
          id={`${idPrefix}-${name}`}
          property={property}
          value={values[name] ?? property.default}
          onChange={(value) => {
            onChange({ ...values, [name]: value });
          }}
        />
      ))}
    </div>
  );
}

interface ControlProps {
  id: string;
  property: Property;
  value: unknown;
  onChange: (value: unknown) => void;
}

function Control({ id, property, value, onChange }: ControlProps) {
  const title = property.title ?? id;
  const labels = property["x-labels"] ?? {};
  // Um número também pode ter nome (a profundidade 0 é "no limit").
  const shown =
    property.type === "boolean"
      ? null
      : (labels[String(value)] ??
        (property.enum !== undefined ? String(value) : formatValue(value)));

  return (
    <div className="flex flex-col gap-2.5 py-3.5 first:pt-0 last:pb-0">
      <div className="flex min-h-6 items-center justify-between gap-3">
        <div className="flex items-center gap-1">
          <Label id={labelId(id)} htmlFor={id} className="text-sm text-soft">
            {title}
          </Label>
          {/* O que o hiperparâmetro faz fica no "?", não à vista (ADR 0032). */}
          {property.description !== undefined && (
            <Explain title={title}>{property.description}</Explain>
          )}
        </div>
        {shown !== null && property.enum === undefined && (
          <span className="font-mono text-sm text-strong tabular-nums">{shown}</span>
        )}
      </div>
      <Field id={id} property={property} value={value} onChange={onChange} />
    </div>
  );
}

function Field({ id, property, value, onChange }: ControlProps) {
  const labels = property["x-labels"] ?? {};
  if (property.enum !== undefined) {
    const options = property.enum.map(String);
    if (options.length <= 4) {
      return (
        <ToggleGroup
          id={id}
          aria-labelledby={labelId(id)}
          type="single"
          variant="outline"
          size="sm"
          value={String(value)}
          onValueChange={(chosen) => {
            if (chosen !== "") {
              onChange(property.enum?.[options.indexOf(chosen)] ?? chosen);
            }
          }}
          className="flex-wrap justify-start"
        >
          {options.map((option) => (
            <ToggleGroupItem key={option} value={option} className="px-3">
              {labels[option] ?? option}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      );
    }
    return (
      <Select
        value={String(value)}
        onValueChange={(chosen) => {
          onChange(property.enum?.[options.indexOf(chosen)] ?? chosen);
        }}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {labels[option] ?? option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }
  if (property.type === "boolean") {
    return <Switch id={id} checked={value === true} onCheckedChange={onChange} />;
  }
  if (property.type === "integer" || property.type === "number") {
    return <NumberField id={id} property={property} value={value} onChange={onChange} />;
  }
  return null;
}

function NumberField({ id, property, value, onChange }: ControlProps) {
  const integer = property.type === "integer";
  const minimum = property.minimum ?? property.exclusiveMinimum ?? 0;
  const maximum = property.maximum ?? property.exclusiveMaximum ?? Math.max(1, minimum * 10);
  const current = clamp(
    typeof value === "number" ? value : Number(property.default ?? minimum),
    minimum,
    maximum,
  );
  const logarithmic =
    property["x-scale"] === "log" ||
    (integer && maximum / Math.max(1, minimum) > 50 && maximum - minimum > 300);

  if (logarithmic) {
    const options = stepOptions(minimum, maximum, current, integer);
    const position = nearestIndex(options, current);
    return (
      <Slider
        id={id}
        aria-labelledby={labelId(id)}
        valueText={formatValue(current)}
        min={0}
        max={options.length - 1}
        step={1}
        value={[position]}
        onValueChange={([index]) => {
          onChange(options[index ?? position] ?? current);
        }}
      />
    );
  }
  const step = integer ? 1 : niceStep(minimum, maximum);
  return (
    <Slider
      id={id}
      aria-labelledby={labelId(id)}
      valueText={formatValue(current)}
      min={minimum}
      max={maximum}
      step={step}
      value={[current]}
      onValueChange={([chosen]) => {
        if (chosen !== undefined) {
          onChange(integer ? Math.round(chosen) : Number(chosen.toPrecision(12)));
        }
      }}
    />
  );
}

/** O id do rótulo de um controlo: os que não são campos de formulário nomeiam-se com ele. */
function labelId(id: string): string {
  return `${id}-rotulo`;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function nearestIndex(options: readonly number[], value: number): number {
  let best = 0;
  options.forEach((option, index) => {
    if (Math.abs(option - value) < Math.abs((options[best] ?? option) - value)) {
      best = index;
    }
  });
  return best;
}
