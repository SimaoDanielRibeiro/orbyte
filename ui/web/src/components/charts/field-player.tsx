/**
 * O campo do modelo com os controlos para rever o treino: reproduzir, pausar e arrastar pelas
 * fotografias. Durante um treino ao vivo, segue a última fotografia que chegou.
 *
 * Com "reduzir movimento" ligado, nada arranca sozinho: mostra-se logo a última fotografia.
 */
import { Pause, Play, RotateCcw } from "lucide-react";
import { useReducedMotion } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";

import type { TrainingAnimation } from "@/api/generated";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { formatCount, formatStatistic } from "@/lib/format";

import { FieldView } from "./field";

/** Quanto tempo leva a reprodução inteira, seja qual for o número de fotografias. */
const PLAYBACK_MS = 4500;

interface FieldPlayerProps {
  animation: TrainingAnimation;
  description: string;
  caption?: ReactNode;
  height?: number;
  /** Começar a reproduzir sozinho, do primeiro passo. */
  autoPlay?: boolean;
  /** Mostrar sempre a fotografia mais recente (o treino ao vivo). */
  follow?: boolean;
  /** Chamado com o passo à vista, para outro gráfico (a rede) mostrar o mesmo passo. */
  onStepChange?: (step: number) => void;
}

export function FieldPlayer({
  animation,
  description,
  caption,
  height,
  autoPlay = false,
  follow = false,
  onStepChange,
}: FieldPlayerProps) {
  const reduced = useReducedMotion() ?? false;
  const count = animation.frames.length;
  const last = count - 1;
  const [index, setIndex] = useState(() => (autoPlay && !reduced ? 0 : last));
  const [playing, setPlaying] = useState(autoPlay && !reduced && count > 1);
  const shown = follow ? last : Math.min(index, last);

  useEffect(() => {
    if (!playing) {
      return;
    }
    const interval = Math.max(40, Math.min(250, PLAYBACK_MS / Math.max(1, count)));
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= last) {
          setPlaying(false);
          return last;
        }
        return current + 1;
      });
    }, interval);
    return () => {
      window.clearInterval(timer);
    };
  }, [playing, count, last]);

  const frame = animation.frames[shown];
  const atEnd = shown >= last;
  const step = frame?.step;

  useEffect(() => {
    if (step !== undefined) {
      onStepChange?.(step);
    }
  }, [step, onStepChange]);

  return (
    <div className="flex flex-col gap-3">
      <FieldView
        animation={animation}
        frameIndex={shown}
        description={description}
        caption={caption}
        height={height}
      />
      {count > 1 && !follow && (
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label={
              playing ? "Pausar o treino" : atEnd ? "Ver o treino outra vez" : "Reproduzir o treino"
            }
            onClick={() => {
              if (playing) {
                setPlaying(false);
                return;
              }
              if (atEnd) {
                setIndex(0);
              }
              setPlaying(true);
            }}
          >
            {playing ? <Pause /> : atEnd ? <RotateCcw /> : <Play />}
          </Button>
          <Slider
            aria-label="Fotografia do treino"
            className="flex-1"
            min={0}
            max={last}
            step={1}
            value={[shown]}
            onValueChange={([value]) => {
              setPlaying(false);
              setIndex(value ?? last);
            }}
          />
          <span className="min-w-36 text-right text-xs text-muted-foreground tabular-nums">
            {frame === undefined ? "" : `Passo ${formatCount(frame.step)}`}
            {frame?.loss !== null && frame?.loss !== undefined
              ? `, loss ${formatStatistic(frame.loss)}`
              : ""}
          </span>
        </div>
      )}
    </div>
  );
}
