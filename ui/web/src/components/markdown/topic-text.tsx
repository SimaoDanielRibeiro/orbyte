import { useTopic } from "@/api/topics";

import { Prose } from "./prose";

interface TopicTextProps {
  /** O nome do tema no backend (`"correlation_matrix"`). */
  topic: string;
  className?: string;
}

/** O texto de um tema, pequeno: a legenda de como se lê um gráfico ou uma tabela. */
export function TopicText({ topic, className }: TopicTextProps) {
  const explanation = useTopic(topic);
  return explanation === undefined ? null : (
    <Prose text={explanation.text} size="small" className={className} />
  );
}
