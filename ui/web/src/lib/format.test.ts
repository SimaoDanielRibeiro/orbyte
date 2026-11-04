/**
 * Os mesmos exemplos de `tests/unit/test_formatting.py` (e das docstrings de
 * `ml_lab/formatting.py`): os dois lados têm de escrever os números da mesma maneira.
 */
import { describe, expect, it } from "vitest";

import {
  formatCount,
  formatList,
  formatMetric,
  formatMoment,
  formatNumber,
  formatOrdinal,
  formatPercent,
  formatQuantity,
  formatScientific,
  formatSignificant,
  formatStatistic,
  formatValue,
} from "./format";

describe("formatNumber", () => {
  it.each([
    [3.14159, 2, "3.14"],
    [0.5, 1, "0.5"],
    [-2.25, 2, "-2.25"],
    [1234.5, 1, "1,234.5"],
    [12345.678, 2, "12,345.68"],
    [-1_000_000, 0, "-1,000,000"],
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
    expect(formatPercent(0.8125)).toBe("81.2%");
  });
});

describe("formatScientific", () => {
  it.each([
    [0.000001, "1 × 10⁻⁶"],
    [0.000025, "2.5 × 10⁻⁵"],
    [1e-10, "1 × 10⁻¹⁰"],
    [-0.00012345, "-1.23 × 10⁻⁴"],
    [5e12, "5 × 10¹²"],
    [0, "0"],
  ])("escreve %d como %s", (value, expected) => {
    expect(formatScientific(value)).toBe(expected);
  });
});

describe("formatSignificant", () => {
  it.each([
    [0.320912, "0.3209"],
    [12.3456, "12.35"],
    [12345.678, "12,346"],
    [4661704537.05, "4.662 × 10⁹"],
    [0.000123456, "1.235 × 10⁻⁴"],
    [-0.5, "-0.5000"],
    [0, "0"],
  ])("escreve %d como %s", (value, expected) => {
    expect(formatSignificant(value)).toBe(expected);
  });
});

describe("contagens, percentagens e quantidades", () => {
  it("escreve contagens e percentagens", () => {
    expect(formatCount(20640)).toBe("20,640");
    expect(formatCount(891)).toBe("891");
    expect(formatPercent(0.1987)).toBe("19.9%");
    expect(formatPercent(1)).toBe("100.0%");
  });

  it("concorda o nome com o número", () => {
    expect(formatQuantity(1, "row", "rows")).toBe("1 row");
    expect(formatQuantity(0, "row", "rows")).toBe("0 rows");
    expect(formatQuantity(20640, "row", "rows")).toBe("20,640 rows");
    expect(formatQuantity(1, "duplicate row", "duplicate rows")).toBe("1 duplicate row");
  });

  it("escreve as posições em inglês", () => {
    expect(formatOrdinal(1)).toBe("1st");
    expect(formatOrdinal(2)).toBe("2nd");
    expect(formatOrdinal(3)).toBe("3rd");
    expect(formatOrdinal(4)).toBe("4th");
    expect(formatOrdinal(11)).toBe("11th");
    expect(formatOrdinal(12)).toBe("12th");
    expect(formatOrdinal(13)).toBe("13th");
    expect(formatOrdinal(21)).toBe("21st");
    expect(formatOrdinal(22)).toBe("22nd");
  });
});

describe("formatValue e formatList", () => {
  it("escreve os valores como se leem em inglês", () => {
    expect(formatValue(0.000714)).toBe("0.000714");
    expect(formatValue(3.0)).toBe("3");
    expect(formatValue(25_000)).toBe("25,000");
    expect(formatValue(false)).toBe("no");
    expect(formatValue(true)).toBe("yes");
    expect(formatValue("rbf")).toBe("rbf");
    expect(formatValue(0.0128)).toBe("0.0128");
    expect(formatValue(0.1)).toBe("0.1");
    expect(formatValue(0.00001)).toBe("1 × 10⁻⁵");
    expect(formatValue(null)).toBe("—");
  });

  it("junta as listas com «and» antes do último", () => {
    expect(formatList([])).toBe("");
    expect(formatList(["Sex"])).toBe("Sex");
    expect(formatList(["Sex", "Embarked"])).toBe("Sex and Embarked");
    expect(formatList(["Sex", "Embarked", "Pclass"])).toBe("Sex, Embarked and Pclass");
  });
});

describe("formatStatistic", () => {
  it("guarda até quatro algarismos significativos, sem zeros à direita", () => {
    expect(formatStatistic(0.42)).toBe("0.42");
    expect(formatStatistic(29.69911764705882)).toBe("29.7");
    expect(formatStatistic(14.526497332334042)).toBe("14.53");
    expect(formatStatistic(80)).toBe("80");
    expect(formatStatistic(20.125)).toBe("20.12");
    expect(formatStatistic(-6.6875)).toBe("-6.688");
  });

  it("escreve por inteiro a partir de mil, e em potência de 10 os muito pequenos", () => {
    expect(formatStatistic(206_855.8)).toBe("206,856");
    expect(formatStatistic(2_635.76)).toBe("2,636");
    expect(formatStatistic(0.0000123)).toBe("1.23 × 10⁻⁵");
    expect(formatStatistic(0)).toBe("0");
  });
});

describe("formatMoment", () => {
  it("escreve o dia e a hora locais, por extenso curto", () => {
    // Sem fuso no texto, o momento lê-se na hora local: o resultado não depende de onde se corre.
    expect(formatMoment("2026-10-24T15:30:00")).toBe("24 Oct 2026, 15:30");
    expect(formatMoment("2026-01-05T09:05:00")).toBe("05 Jan 2026, 09:05");
    expect(formatMoment("yesterday")).toBe("—");
  });
});

describe("formatMetric", () => {
  it("escreve cada métrica como o MetricResult.formatted do Python", () => {
    expect(formatMetric({ value: 0.9667, as_percent: true, higher_is_better: true })).toBe("96.7%");
    expect(formatMetric({ value: 0.1234, as_percent: false, higher_is_better: false })).toBe(
      "0.123",
    );
    expect(formatMetric({ value: 3, as_percent: false, higher_is_better: null })).toBe("3");
    expect(formatMetric({ value: null, as_percent: true, higher_is_better: true })).toBe("—");
  });
});
