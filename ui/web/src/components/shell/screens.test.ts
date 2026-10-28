import { describe, expect, it } from "vitest";

import { belongsTo, interfaceOf, isCurrent, NAVIGATION, SCREENS } from "./screens";

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
      const paths = NAVIGATION[path].flatMap((group) => group.screens.map((screen) => screen.to));

      expect(new Set(paths).size).toBe(paths.length);
    },
  );

  it("os dois fluxos começam nos dados e acabam a explicar o run", () => {
    for (const groups of Object.values(NAVIGATION)) {
      const flow = groups[0]?.screens ?? [];
      expect(flow[0]).toBe(SCREENS.data);
      expect(flow.at(-1)).toBe(SCREENS.explain);
    }
  });

  it("um ecrã que só existe numa interface leva para ela, e os partilhados não mudam nada", () => {
    expect(interfaceOf("/preparar")).toBe("manual");
    expect(interfaceOf("/rl/run-1a2b")).toBe("manual");
    expect(interfaceOf("/agentes/search-1a2b")).toBe("agents");
    expect(interfaceOf("/dados")).toBeNull();
    expect(interfaceOf("/treinar/run-1a2b")).toBeNull();
  });

  it("um ecrã de uma só interface está só na barra dela", () => {
    expect(belongsTo("/preparar", "manual")).toBe(true);
    expect(belongsTo("/preparar", "agents")).toBe(false);
    expect(belongsTo("/agentes", "agents")).toBe(true);
    expect(belongsTo("/agentes", "manual")).toBe(false);
  });

  it("todos os ecrãs, menos o início, estão numa das barras", () => {
    for (const screen of Object.values(SCREENS)) {
      if (screen.to !== "/") {
        expect(belongsTo(screen.to, "manual") || belongsTo(screen.to, "agents")).toBe(true);
      }
    }
  });
});
