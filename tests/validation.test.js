import { test } from "node:test";
import assert from "node:assert/strict";
import { numericError } from "../src/validation.js";
test("epochs reject blanks, fractions and out-of-range values", () => {
  const input = (value) => ({ value, min: "1", max: "30", dataset: {} });
  for (const value of ["", " ", "0", "-1", "31", "1.5", "NaN", "Infinity"])
    assert.ok(numericError(input(value)), value);
  for (const value of ["1", "3", "30"])
    assert.equal(numericError(input(value)), "");
});
test("layer ranges include valid decimal dropout and integer-only neuron counts", () => {
  const input = (value, key, min, max) => ({
    value,
    dataset: { key },
    min,
    max,
  });
  assert.equal(numericError(input("0.25", "rate", "0", "0.8")), "");
  assert.equal(numericError(input("0", "rate", "0", "0.8")), "");
  assert.ok(numericError(input("", "rate", "0", "0.8")));
  assert.ok(numericError(input("0.81", "rate", "0", "0.8")));
  assert.ok(numericError(input("3.5", "filters", "1", "64")));
  assert.ok(numericError(input("257", "units", "1", "256")));
  assert.equal(numericError(input("256", "units", "1", "256")), "");
});
