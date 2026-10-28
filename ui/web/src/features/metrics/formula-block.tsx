/** A fórmula de uma métrica (KaTeX) e o que quer dizer cada símbolo dela. */
import type { MetricCard } from "@/api/generated";
import { Formula, Prose } from "@/components/markdown/prose";

export function FormulaBlock({ card }: { card: MetricCard }) {
  return (
    // `min-w-0`: numa grelha, uma fórmula larga desliza na sua caixa em vez de alargar a página.
    <section
      className="flex min-w-0 flex-col gap-3"
      aria-label={`A fórmula de ${card.display_name}`}
    >
      <div className="relative overflow-x-auto rounded-xl border bg-ground px-4 py-5 text-strong">
        <Formula tex={card.formula} />
      </div>
      {card.symbols.length > 0 && (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
          {card.symbols.map((symbol) => (
            <div key={symbol.symbol} className="contents">
              <dt className="text-right">
                <Formula tex={symbol.symbol} display={false} />
              </dt>
              <dd>
                {/* O significado pode ter LaTeX ("a classe certa do exemplo $i$"). */}
                <Prose text={symbol.meaning} size="small" />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
