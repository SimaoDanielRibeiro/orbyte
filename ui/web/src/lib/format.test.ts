/**
 * Os mesmos exemplos de `tests/unit/test_formatting.py` (e das docstrings de
 * `ml_lab/formatting.py`): os dois lados têm de escrever os números da mesma maneira.
 */
import { describe, expect, it } from "vitest";

import {
  formatCount,
  formatList,
  formatMetric,
  formatNumber,
  formatPercent,
  formatQuantity,
  formatScientific,
  formatSignificant,
  formatStatistic,
  formatValue,
} from "./format";

describe("formatNumber", () => {
  it.each([
    [3.14159, 2, "3,14"],
    [0.5, 1, "0,5"],
    [-2.25, 2, "-2,25"],
    [1234.5, 1, "1234,5"],
    [12345.678, 2, "12 345,68"],
    [-1_000_000, 0, "-1 000 000"],
  ])("escreve %d com %d casas como %s", (value, decimals, expected) => {
    expect(formatNumber(value, decimals)).toBe(expected);
  });

  it("dá um símbolo próprio aos números não finitos", () => {
    expect(formatNumber(Number.NaN)).toBe("—");
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("∞");
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe("-∞");
  });

  it("arredonda metade para o par, como o Python", () => {
    expect(formatNumber(2.5, 0)).toBe("2");
    expect(formatNumber(3.5, 0)).toBe("4");
    expect(formatPercent(0.8125)).toBe("81,2%");
  });
});

describe("formatScientific", () => {
  it.each([
    [0.000001, "1 × 10⁻⁶"],
    [0.000025, "2,5 × 10⁻⁵"],
    [1e-10, "1 × 10⁻¹⁰"],
    [-0.00012345, "-1,23 × 10⁻⁴"],
    [5e12, "5 × 10¹²"],
    [0, "0"],
  ])("escreve %d como %s", (value, expected) => {
    expect(formatScientific(value)).toBe(expected);
  });
});

describe("formatSignificant", () => {
  it.each([
    [0.320912, "0,3209"],
    [12.3456, "12,35"],
    [12345.678, "12 346"],
    [4661704537.05, "4,662 × 10⁹"],
    [0.000123456, "1,235 × 10⁻⁴"],
    [-0.5, "-0,5000"],
    [0, "0"],
  ])("escreve %d como %s", (value, expected) => {
    expect(formatSignificant(value)).toBe(expected);
  });
});

describe("contagens, percentagens e quantidades", () => {
  it("escreve contagens e percentagens", () => {
    expect(formatCount(20640)).toBe("20 640");
    expect(formatCount(891)).toBe("891");
    expect(formatPercent(0.1987)).toBe("19,9%");
    expect(formatPercent(1)).toBe("100,0%");
  });

  it("concorda o nome com o número", () => {
    expect(formatQuantity(1, "linha", "linhas")).toBe("1 linha");
    expect(formatQuantity(0, "linha", "linhas")).toBe("0 linhas");
    expect(formatQuantity(20640, "linha", "linhas")).toBe("20 640 linhas");
    expect(formatQuantity(1, "linha duplicada", "linhas duplicadas")).toBe("1 linha duplicada");
  });
});

describe("formatValue e formatList", () => {
  it("escreve os valores como se lê em português", () => {
    expect(formatValue(0.000714)).toBe("0,000714");
    expect(formatValue(3.0)).toBe("3");
    expect(formatValue(25_000)).toBe("25 000");
    expect(formatValue(false)).toBe("não");
    expect(formatValue(true)).toBe("sim");
    expect(formatValue("rbf")).toBe("rbf");
    expect(formatValue(0.0128)).toBe("0,0128");
    expect(formatValue(0.1)).toBe("0,1");
    expect(formatValue(0.00001)).toBe("1 × 10⁻⁵");
    expect(formatValue(null)).toBe("—");
  });

  it("junta as listas com «e» antes do último", () => {
    expect(formatList([])).toBe("");
    expect(formatList(["Sex"])).toBe("Sex");
    expect(formatList(["Sex", "Embarked"])).toBe("Sex e Embarked");
    expect(formatList(["Sex", "Embarked", "Pclass"])).toBe("Sex, Embarked e Pclass");
  });
});

describe("formatStatistic", () => {
  it("guarda até quatro algarismos significativos, sem zeros à direita", () => {
    expect(formatStatistic(0.42)).toBe("0,42");
    expect(formatStatistic(29.69911764705882)).toBe("29,7");
    expect(formatStatistic(14.526497332334042)).toBe("14,53");
    expect(formatStatistic(80)).toBe("80");
    expect(formatStatistic(20.125)).toBe("20,12");
    expect(formatStatistic(-6.6875)).toBe("-6,688");
  });

  it("escreve por inteiro a partir de mil, e em potência de 10 os muito pequenos", () => {
    expect(formatStatistic(206_855.8)).toBe("206 856");
    expect(formatStatistic(2_635.76)).toBe("2636");
    expect(formatStatistic(0.0000123)).toBe("1,23 × 10⁻⁵");
    expect(formatStatistic(0)).toBe("0");
  });
});

describe("formatMetric", () => {
  it("escreve cada métrica como o MetricResult.formatted do Python", () => {
    expect(formatMetric({ value: 0.9667, as_percent: true, higher_is_better: true })).toBe("96,7%");
    expect(formatMetric({ value: 0.1234, as_percent: false, higher_is_better: false })).toBe(
      "0,123",
    );
    expect(formatMetric({ value: 3, as_percent: false, higher_is_better: null })).toBe("3");
    expect(formatMetric({ value: null, as_percent: true, higher_is_better: true })).toBe("—");
  });
});
