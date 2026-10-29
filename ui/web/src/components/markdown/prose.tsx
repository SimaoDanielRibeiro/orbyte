/**
 * Os textos didáticos do backend, em Markdown, com as fórmulas em LaTeX (`$…$` e `$$…$$`)
 * desenhadas pelo KaTeX: as fichas, os temas do "Porquê?", as explicações das métricas.
 *
 * Na mesma Geist do resto, com a entrelinha mais generosa e linhas curtas (ADR 0032).
 */
import "katex/dist/katex.min.css";

import { cn } from "cn";
import katex from "katex";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

const COMPONENTS: Components = {
  h1: ({ children }) => <h2 className="mt-8 mb-3 text-xl">{children}</h2>,
  h2: ({ children }) => <h3 className="mt-6 mb-2 text-lg">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-5 mb-2 text-sm font-medium text-strong">{children}</h4>,
  p: ({ children }) => <p className="my-3">{children}</p>,
  ul: ({ children }) => (
    <ul className="my-3 list-disc space-y-1.5 pl-5 marker:text-muted-foreground">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-3 list-decimal space-y-1.5 pl-5 marker:text-muted-foreground">{children}</ol>
  ),
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  a: ({ children, href }) => (
    <a
      href={href}
      className="underline decoration-input underline-offset-4 hover:decoration-foreground"
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.9em]">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="relative my-4 overflow-x-auto rounded-lg border bg-card p-3 text-sm">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-4 border-l-2 border-input pl-4 text-muted-foreground">
      {children}
    </blockquote>
  ),
  table: ({ children }) => (
    <div className="relative my-4 overflow-x-auto">
      <table className="w-full border-collapse font-sans text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border-b px-3 py-2 text-left font-medium">{children}</th>,
  td: ({ children }) => (
    <td className="border-b border-border/60 px-3 py-2 align-top">{children}</td>
  ),
};

interface ProseProps {
  text: string;
  /** `reading` para o texto corrido; `small` para as legendas por baixo dos gráficos. */
  size?: "reading" | "small";
  /**
   * As extensões do GitHub (as tabelas): a ficha de um modelo tem uma tabela das métricas. Os
   * textos didáticos não as usam, e um til sozinho ("~0,5") continua a ser só um til.
   */
  gfm?: boolean;
  className?: string;
}

export function Prose({ text, size = "reading", gfm = false, className }: ProseProps) {
  return (
    <div
      className={cn(
        "max-w-[68ch]",
        size === "reading"
          ? "text-reading text-foreground/90"
          : "text-[0.8125rem] leading-relaxed text-soft [&_p]:my-0 [&_p+p]:mt-2",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={gfm ? [[remarkGfm, { singleTilde: false }], remarkMath] : [remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={COMPONENTS}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

interface FormulaProps {
  /** A fórmula, em LaTeX (vem do backend). */
  tex: string;
  /** Em bloco, ao centro, como num livro; senão, no meio do texto. */
  display?: boolean;
  className?: string;
}

/** Uma fórmula sozinha (a de uma métrica). */
export function Formula({ tex, display = true, className }: FormulaProps) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
    output: "htmlAndMathml",
  });
  return (
    <span
      className={cn(
        // `relative` nas duas: o MathML que o KaTeX junta para os leitores de ecrã tem posição
        // absoluta, e sem um contentor posicionado alargava a página no telemóvel.
        display ? "relative block overflow-x-auto py-1 text-center text-lg" : "relative inline",
        className,
      )}
      // O LaTeX vem do backend do próprio laboratório e o KaTeX escapa o texto.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
