export function numericError(input) {
  const name =
    input.dataset.key === "units"
      ? "Neurons"
      : input.dataset.key === "filters"
        ? "Filters"
        : input.dataset.key === "rate"
          ? "Drop rate"
          : "Epochs";
  const raw = input.value.trim(),
    n = Number(raw);
  const integer = input.dataset.key !== "rate";
  if (
    !raw ||
    !Number.isFinite(n) ||
    n < Number(input.min) ||
    n > Number(input.max) ||
    (integer && !Number.isInteger(n))
  ) {
    return `${name}: enter ${integer ? "a whole number" : "a number"} from ${input.min} to ${input.max}.`;
  }
  return "";
}
