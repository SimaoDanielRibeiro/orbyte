import { describe, expect, it } from "vitest";

import {
  belongsTo,
  FIRST_STEPS,
  interfaceOf,
  isCurrent,
  landingOf,
  NAVIGATION,
  PINNED,
  SCREENS,
  stepsOf,
} from "./screens";

describe("landingOf", () => {
  it("com os dados abertos, a troca leva ao primeiro passo da outra interface", () => {
    expect(landingOf("agents", true)).toBe("/agents");
    expect(landingOf("manual", true)).toBe("/prepare");
  });

  it("sem dados, a troca leva aos Dados, onde as duas começam", () => {
    expect(landingOf("agents", false)).toBe("/data");
    expect(landingOf("manual", false)).toBe("/data");
  });
});

describe("isCurrent", () => {
  it("o início só é o atual na própria página", () => {
    expect(isCurrent("/", "/")).toBe(true);
    expect(isCurrent("/", "/data")).toBe(false);
  });

  it("um ecrã é o atual nas páginas dentro dele, mas não nas que só começam igual", () => {
    expect(isCurrent("/models", "/models")).toBe(true);
    expect(isCurrent("/models", "/models/knn")).toBe(true);
    expect(isCurrent("/train", "/train/run-1a2b")).toBe(true);
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
        ...PINNED.map((screen) => screen.to),
      ];

      expect(new Set(paths).size).toBe(paths.length);
    },
  );

  it("os dois fluxos começam nos dados, passam pela exploração e acabam a guardar o modelo", () => {
    for (const path of ["manual", "agents"] as const) {
      const steps = stepsOf(path);
      expect(steps.slice(0, 2)).toEqual([SCREENS.data, SCREENS.explore]);
      expect(steps.slice(-2)).toEqual([SCREENS.explain, SCREENS.save]);
    }
  });

  it("um ecrã que só existe numa interface leva para ela, e os partilhados não mudam nada", () => {
    expect(interfaceOf("/prepare")).toBe("manual");
    expect(interfaceOf("/rl/run-1a2b")).toBe("manual");
    expect(interfaceOf("/agents/search-1a2b")).toBe("agents");
    expect(interfaceOf("/data")).toBeNull();
    expect(interfaceOf("/explore")).toBeNull();
    expect(interfaceOf("/train/run-1a2b")).toBeNull();
    expect(interfaceOf("/history")).toBeNull();
  });

  it("um ecrã de uma só interface está só na barra dela, e os partilhados nas duas", () => {
    expect(belongsTo("/prepare", "manual")).toBe(true);
    expect(belongsTo("/prepare", "agents")).toBe(false);
    expect(belongsTo("/agents", "agents")).toBe(true);
    expect(belongsTo("/agents", "manual")).toBe(false);
    expect(belongsTo("/explore", "manual")).toBe(true);
    expect(belongsTo("/explore", "agents")).toBe(true);
    expect(belongsTo("/history", "manual")).toBe(true);
    expect(belongsTo("/history", "agents")).toBe(true);
  });

  it("todos os ecrãs, menos o início, estão numa das barras", () => {
    for (const screen of Object.values(SCREENS)) {
      if (screen.to !== "/") {
        expect(belongsTo(screen.to, "manual") || belongsTo(screen.to, "agents")).toBe(true);
      }
    }
  });
});
