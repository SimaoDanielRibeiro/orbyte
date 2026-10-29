import { describe, expect, it } from "vitest";

import { parseCsv } from "./csv";

describe("parseCsv", () => {
  it("separa as linhas e os campos", () => {
    expect(parseCsv("Age,Sex\n22,male\n38,female\n")).toEqual([
      ["Age", "Sex"],
      ["22", "male"],
      ["38", "female"],
    ]);
  });

  it("guarda as vírgulas, as quebras de linha e as aspas de um campo entre aspas", () => {
    expect(
      parseCsv('Name,previsto\r\n"Braund, Mr. Owen ""Harris""",0\r\n"duas\nlinhas",1'),
    ).toEqual([
      ["Name", "previsto"],
      ['Braund, Mr. Owen "Harris"', "0"],
      ["duas\nlinhas", "1"],
    ]);
  });

  it("um campo vazio continua a ser um campo", () => {
    expect(parseCsv("a,b,c\n1,,3")).toEqual([
      ["a", "b", "c"],
      ["1", "", "3"],
    ]);
  });
});
