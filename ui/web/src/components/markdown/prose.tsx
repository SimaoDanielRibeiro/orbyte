/**
 * Os textos didáticos do backend, em Markdown, com as fórmulas em LaTeX (`$…$` e `$$…$$`)
 * desenhadas pelo KaTeX: as fichas, os temas do "Porquê?", as explicações das métricas.
 *
 * É a voz do livro: serifada, com a entrelinha generosa e linhas curtas (ADR 0029).
 */
import "katex/dist/katex.min.css";

import { cn } from "cn";
import katex from "katex";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

const COMPONENTS: Components = {
  h1: ({ children }) => <h2 className="mt-8 mb-3 text-2xl">{children}</h2>,
  h2: ({ children }) => <h3 className="mt-6 mb-2 text-xl">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-5 mb-2 font-sans text-base font-medium">{children}</h4>,
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
    <pre className="my-4 overflow-x-auto rounded-lg border bg-card p-3 text-sm">{children}</pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-4 border-l-2 border-input pl-4 text-muted-foreground">
      {children}
    </blockquote>
  ),
  table: ({ children }) => (
    <div className="my-4 overflow-x-auto">
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
  className?: string;
}

export function Prose({ text, size = "reading", className }: ProseProps) {
  return (
    <div
      className={cn(
        "max-w-[68ch] font-serif",
        size === "reading"
          ? "text-reading text-foreground/90"
          : "text-sm leading-relaxed text-muted-foreground [&_p]:my-0 [&_p+p]:mt-2",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
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
        display ? "block overflow-x-auto py-1 text-center text-lg" : "inline",
        className,
      )}
      // O LaTeX vem do backend do próprio laboratório e o KaTeX escapa o texto.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
