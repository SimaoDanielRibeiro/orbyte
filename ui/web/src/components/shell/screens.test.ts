import { describe, expect, it } from "vitest";

import {
  belongsTo,
  FIRST_STEPS,
  interfaceOf,
  isCurrent,
  NAVIGATION,
  SCREENS,
  stepsOf,
} from "./screens";

describe("isCurrent", () => {
  it("o início só é o atual na própria página", () => {
    expect(isCurrent("/", "/")).toBe(true);
    expect(isCurrent("/", "/dados")).toBe(false);
  });

  it("um ecrã é o atual nas páginas dentro dele, mas não nas que só começam igual", () => {
    expect(isCurrent("/modelos", "/modelos")).toBe(true);
    expect(isCurrent("/modelos", "/modelos/knn")).toBe(true);
    expect(isCurrent("/treinar", "/treinar/run-1a2b")).toBe(true);
    expect(isCurrent("/rl", "/rlx")).toBe(false);
  });
});

describe("as duas interfaces", () => {
  it.each(["manual", "agents"] as const)(
    "na interface %s, cada ecrã aparece uma só vez",
    (path) => {
      const paths = [
        ...FIRST_STEPS.map((screen) => screen.to),
        ...NAVIGATION[path].flatMap((group) => group.screens.map((screen) => screen.to)),
      ];

      expect(new Set(paths).size).toBe(paths.length);
    },
  );

  it("os dois fluxos começam nos dados, passam pela exploração e acabam a explicar o run", () => {
    for (const path of ["manual", "agents"] as const) {
      const steps = stepsOf(path);
      expect(steps.slice(0, 2)).toEqual([SCREENS.data, SCREENS.explore]);
      expect(steps.at(-1)).toBe(SCREENS.explain);
    }
  });

  it("um ecrã que só existe numa interface leva para ela, e os partilhados não mudam nada", () => {
    expect(interfaceOf("/preparar")).toBe("manual");
    expect(interfaceOf("/rl/run-1a2b")).toBe("manual");
    expect(interfaceOf("/agentes/search-1a2b")).toBe("agents");
    expect(interfaceOf("/dados")).toBeNull();
    expect(interfaceOf("/explorar")).toBeNull();
    expect(interfaceOf("/treinar/run-1a2b")).toBeNull();
  });

  it("um ecrã de uma só interface está só na barra dela, e os primeiros passos nas duas", () => {
    expect(belongsTo("/preparar", "manual")).toBe(true);
    expect(belongsTo("/preparar", "agents")).toBe(false);
    expect(belongsTo("/agentes", "agents")).toBe(true);
    expect(belongsTo("/agentes", "manual")).toBe(false);
    expect(belongsTo("/explorar", "manual")).toBe(true);
    expect(belongsTo("/explorar", "agents")).toBe(true);
  });

  it("todos os ecrãs, menos o início, estão numa das barras", () => {
    for (const screen of Object.values(SCREENS)) {
      if (screen.to !== "/") {
        expect(belongsTo(screen.to, "manual") || belongsTo(screen.to, "agents")).toBe(true);
      }
    }
  });
});
