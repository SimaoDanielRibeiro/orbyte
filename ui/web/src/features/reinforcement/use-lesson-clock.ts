/**
 * O relógio da lição do ecrã Reinforcement learning: em que episódio, passo e fase se está, e os
 * controlos (reproduzir, pausar, a velocidade e saltar para um episódio).
 *
 * Cada passo tem quatro fases (o estado, a ação, a recompensa e a correção). No fim de cada
 * episódio há uma pausa, para se ler o que aconteceu, e depois segue-se para o próximo; no fim
 * do último, a lição pára. O relógio só anda com a lição à vista.
 */
import { useEffect, useState } from "react";

import type { LessonEpisode } from "@/api/generated";

export type Speed = 1 | 2 | 4;

/** Quanto dura cada fase de um passo, à velocidade 1. */
const PHASE_MS = 380;

/** A pausa no fim de cada episódio, à velocidade 1 (para ler a frase do episódio). */
const HOLD_MS = 3000;

const PHASES = 4;

export interface LessonClock {
  /** O episódio aberto (a posição dele entre os guardados). */
  episode: number;
  /** O passo de agora, nesse episódio. */
  step: number;
  /** A fase de agora: 0, o estado; 1, a ação; 2, a recompensa; 3, a correção. */
  phase: number;
  /** Se o episódio aberto já acabou. */
  finished: boolean;
  /** Se a lição toda já acabou. */
  atEnd: boolean;
  playing: boolean;
  speed: Speed;
  toggle: () => void;
  setSpeed: (speed: Speed) => void;
  goToEpisode: (index: number) => void;
}

interface ClockOptions {
  /** Se a lição está à vista (fora dela, o relógio pára). */
  visible: boolean;
  /** Se começa sozinha (não, com "reduzir movimento"). */
  autoPlay: boolean;
}

export function useLessonClock(
  kept: readonly LessonEpisode[],
  { visible, autoPlay }: ClockOptions,
): LessonClock {
  const [episode, setEpisode] = useState(0);
  const [tick, setTick] = useState(0);
  // O que quem vê pediu (reproduzir ou pausar); enquanto não pedir nada, vale o `autoPlay`.
  const [wanted, setWanted] = useState<boolean | null>(null);
  const [speed, setSpeed] = useState<Speed>(2);
  const total = (kept[episode]?.steps.length ?? 0) * PHASES;
  const finished = tick >= total;
  const atEnd = finished && episode >= kept.length - 1;
  const playing = (wanted ?? autoPlay) && !atEnd;

  useEffect(() => {
    if (!playing || !visible) {
      return;
    }
    const timer = window.setTimeout(
      () => {
        if (!finished) {
          setTick((value) => value + 1);
        } else {
          setEpisode((value) => value + 1);
          setTick(0);
        }
      },
      (finished ? HOLD_MS : PHASE_MS) / speed,
    );
    return () => {
      window.clearTimeout(timer);
    };
  }, [playing, visible, finished, tick, speed]);

  const position = Math.min(tick, Math.max(0, total - 1));
  return {
    episode,
    step: Math.floor(position / PHASES),
    phase: finished ? PHASES - 1 : tick % PHASES,
    finished,
    atEnd,
    playing,
    speed,
    toggle: () => {
      if (atEnd) {
        setEpisode(0);
        setTick(0);
        setWanted(true);
        return;
      }
      setWanted(!playing);
    },
    setSpeed,
    goToEpisode: (index: number) => {
      setEpisode(index);
      setTick(0);
    },
  };
}
