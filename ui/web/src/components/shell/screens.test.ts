import { describe, expect, it } from "vitest";

import { isCurrent, LEARN_SCREENS, OTHER_SCREENS, PATH_STEPS, SCREENS } from "./screens";

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

describe("a barra lateral", () => {
  it.each(["manual", "agents"] as const)("no caminho %s, cada ecrã aparece uma só vez", (path) => {
    const shown = [SCREENS.home, ...PATH_STEPS[path], ...LEARN_SCREENS[path], ...OTHER_SCREENS];
    const paths = shown.map((screen) => screen.to);

    expect(new Set(paths).size).toBe(paths.length);
  });

  it("os dois caminhos começam nos dados e acabam a explicar o run", () => {
    for (const steps of Object.values(PATH_STEPS)) {
      expect(steps[0]).toBe(SCREENS.data);
      expect(steps.at(-1)).toBe(SCREENS.explain);
    }
  });
});
