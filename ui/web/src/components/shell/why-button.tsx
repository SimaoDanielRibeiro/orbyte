/**
 * O "?" com as ideias por trás de um ecrã, para os cabeçalhos feitos à mão (o do treino, o de
 * um modelo, o de um agente). Os outros ecrãs dão os temas ao `PageHeader`, que mostra o mesmo.
 */
import { PageExplain } from "@/components/explain/explain";

interface WhyButtonProps {
  /** Os temas didáticos que o backend escreve uma só vez (regra 5 do `CONTRIBUTING.md`). */
  topics: readonly string[];
  /** O nome do ecrã (o título do painel que abre). */
  title?: string;
  /** O que se faz no ecrã, numa frase. */
  description?: string;
}

export function WhyButton({ topics, title = "Why?", description }: WhyButtonProps) {
  return <PageExplain title={title} description={description} topics={topics} />;
}
