import { cn } from "cn";
import { CircleAlert, Info, TriangleAlert, type LucideIcon } from "lucide-react";

import type { Notice, NoticeLevel } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";

const LEVEL_STYLES: Readonly<Record<NoticeLevel, { icon: LucideIcon; className: string }>> = {
  info: { icon: Info, className: "text-muted-foreground" },
  warning: { icon: TriangleAlert, className: "text-now" },
  error: { icon: CircleAlert, className: "text-destructive" },
};

/**
 * Os avisos do backend (`Notice`), pela ordem em que vêm (os erros primeiro). A mensagem já
 * está escrita para quem a lê; o nível dá o ícone e a cor, e o nome do nível vai para os
 * leitores de ecrã.
 */
export function NoticeList({
  notices,
  className,
}: {
  notices: readonly Notice[];
  className?: string;
}) {
  const label = useLabel();
  if (notices.length === 0) {
    return null;
  }
  return (
    <ul className={cn("flex flex-col gap-3", className)}>
      {notices.map((notice, index) => {
        const { icon: Icon, className: tone } = LEVEL_STYLES[notice.level];
        return (
          <li key={`${notice.code}-${String(index)}`} className="flex gap-3">
            <Icon aria-hidden="true" className={cn("mt-0.5 size-4 shrink-0", tone)} />
            <p className="max-w-[68ch] font-serif text-reading">
              <span className="sr-only">{label("NoticeLevel", notice.level)}: </span>
              {notice.message}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
