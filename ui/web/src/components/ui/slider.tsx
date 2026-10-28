"use client";

import * as React from "react";
import { cn } from "cn";
import { Slider as SliderPrimitive } from "radix-ui";

/**
 * O nome e o valor escrito vão para os "polegares" (o elemento com o foco e o papel `slider`),
 * não para a raiz: um `<label for>` não dá nome a um `span`, por isso quem usa o controlo passa
 * `aria-labelledby` (o id do rótulo) ou `aria-label`, e `valueText` com o valor como se lê
 * (num controlo em passos, o valor interno é a posição, não o número).
 */
function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  valueText,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root> & { valueText?: string }) {
  const _values = React.useMemo(
    () => (Array.isArray(value) ? value : Array.isArray(defaultValue) ? defaultValue : [min, max]),
    [value, defaultValue, min, max],
  );

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "relative flex w-full touch-none items-center select-none data-disabled:opacity-50 data-vertical:h-full data-vertical:min-h-40 data-vertical:w-auto data-vertical:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="relative grow overflow-hidden rounded-full bg-[#2e2e2e] data-horizontal:h-1.5 data-horizontal:w-full data-vertical:h-full data-vertical:w-1.5"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className="absolute bg-primary select-none data-horizontal:h-full data-vertical:w-full"
        />
      </SliderPrimitive.Track>
      {Array.from({ length: _values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          aria-label={ariaLabel}
          aria-labelledby={ariaLabelledBy}
          aria-valuetext={valueText}
          className="block size-4 shrink-0 rounded-full border-2 border-[#121212] bg-white shadow-[0_0_0_1px_#4a4a4a] ring-ring/50 transition-[color,box-shadow] select-none hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50"
        />
      ))}
    </SliderPrimitive.Root>
  );
}

export { Slider };
