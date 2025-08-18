import React, { useMemo } from "react";

export default function DatabotCoach({ before, after, pipeline, options }) {
  const msgs = useMemo(() => {
    const out = [];
    if (!pipeline || pipeline.length === 0) {
      out.push("No cleaning steps selected yet. Pick a few and hit Preview.");
      return out;
    }

    // Summarize pipeline choices
    const stepLine = (s) => {
      switch (s.op) {
        case "lowercase_headers":
          return "Lowercase headers";
        case "remove_duplicates":
          return "Remove duplicate rows";
        case "conversions":
          return `Convert dtypes → ${s.columns
            .map((c) => `${c.name}→${c.to}`)
            .join(", ")}`;
        case "binning":
          return `Bin → ${s.columns
            .map((c) => `${c.name}:${c.bins}`)
            .join(", ")}`;
        case "fillna":
          return `Fill NA (${s.strategy}) → ${s.columns.join(", ")}`;
        case "dropna":
          return "Drop NA rows";
        case "encoding":
          return `Encode (${s.method}) → ${s.columns.join(", ")}`;
        case "outliers":
          return `Outliers (${s.method}) → ${s.columns.join(", ")}`;
        case "scale":
          return `Scale (${s.method}) → ${s.columns.join(", ")}`;
        default:
          return s.op;
      }
    };
    out.push("Planned steps:");
    out.push(...pipeline.map(stepLine));

    // Preview delta (if available)
    if (before && after && before.shape && after.shape) {
      const [br, bc] = before.shape,
        [ar, ac] = after.shape;
      out.push(`Preview: shape ${br}×${bc} → ${ar}×${ac}`);
      const nb = before.null_counts || {},
        na = after.null_counts || {};
      const improved = Object.keys(na).filter(
        (k) => (nb[k] ?? 0) > (na[k] ?? 0)
      );
      if (improved.length)
        out.push(
          `Nulls reduced in: ${improved.slice(0, 6).join(", ")}${
            improved.length > 6 ? "…" : ""
          }`
        );
    }

    // Quick sanity checks
    const picked = (k) =>
      Array.isArray(options?.selected_columns?.[k]) &&
      options.selected_columns[k].length > 0;
    if (options?.fillna_strategy && !picked("fillna"))
      out.push(
        "Heads-up: imputation strategy is set but no columns were selected."
      );
    if (options?.encoding && !picked("encoding"))
      out.push("Heads-up: encoding method set but no columns were selected.");
    if (options?.scale && !picked("scale"))
      out.push("Heads-up: scaling method set but no columns were selected.");

    return out;
  }, [before, after, pipeline, options]);

  return (
    <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
      <p className="text-sm font-medium text-blue-800">Tracker</p>
      <ul className="mt-1 text-xs text-blue-900 list-disc list-inside space-y-1">
        {msgs.map((m, i) => (
          <li key={i} className="whitespace-pre-wrap">
            {m}
          </li>
        ))}
      </ul>
    </div>
  );
}
