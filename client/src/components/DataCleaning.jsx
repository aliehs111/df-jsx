// client/src/components/DataCleaning.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { debounce } from "lodash"; // Ensure lodash is installed: npm install lodash
import DatabotCoach from "./DatabotCoach";

// Order for display and server execution
const STEP_ORDER = [
  "lowercase_headers",
  "remove_duplicates",
  "conversions",
  "binning",
  "fillna",
  "dropna",
  "encoding",
  "outliers",
  "scale",
];

/* ---------- UI Primitives ---------- */

function Section({ title, subtitle, children, defaultOpen = true }) {
  return (
    <details className="rounded-lg border bg-white" open={defaultOpen}>
      <summary className="cursor-pointer select-none px-4 py-2 text-sm font-semibold bg-gray-50 rounded-t-lg">
        {title}
        {subtitle && (
          <span className="ml-2 text-xs font-normal text-gray-500">
            {subtitle}
          </span>
        )}
      </summary>
      <div className="p-4">{children}</div>
    </details>
  );
}

function ColumnPicker({ columns, value, onChange, sizeHint = "m" }) {
  const [q, setQ] = useState("");
  const selected = new Set(value || []);
  const filtered = useMemo(() => {
    if (!q) return columns;
    const needle = q.toLowerCase();
    return columns.filter((c) => String(c).toLowerCase().includes(needle));
  }, [columns, q]);

  const toggle = (col) => {
    const next = new Set(selected);
    if (next.has(col)) next.delete(col);
    else next.add(col);
    onChange(Array.from(next));
  };

  const selectVisible = () =>
    onChange(Array.from(new Set([...(value || []), ...filtered])));
  const clearVisible = () =>
    onChange((value || []).filter((c) => !filtered.includes(c)));
  const selectAll = () => onChange([...columns]);
  const clearAll = () => onChange([]);

  const maxH =
    sizeHint === "s" ? "max-h-32" : sizeHint === "l" ? "max-h-80" : "max-h-56";

  return (
    <div className="rounded border p-2 bg-white">
      <div className="flex items-center gap-2 mb-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Filter columns…"
          className="w-full rounded border px-2 py-1 text-sm"
        />
        <button
          onClick={selectVisible}
          className="text-xs px-2 py-1 rounded border"
        >
          Select visible
        </button>
        <button
          onClick={clearVisible}
          className="text-xs px-2 py-1 rounded border"
        >
          Clear visible
        </button>
        <button
          onClick={selectAll}
          className="text-xs px-2 py-1 rounded border"
        >
          All
        </button>
        <button onClick={clearAll} className="text-xs px-2 py-1 rounded border">
          None
        </button>
      </div>
      <div className={`overflow-y-auto ${maxH} pr-1`}>
        {filtered.length === 0 ? (
          <div className="text-xs text-gray-500 px-1 py-1">No columns</div>
        ) : (
          <ul className="space-y-1 text-sm">
            {filtered.map((col) => (
              <li key={col} className="flex items-center">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selected.has(col)}
                    onChange={() => toggle(col)}
                  />
                  <span className="font-mono text-xs">{col}</span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </div>
      {value?.length ? (
        <div className="mt-2 text-xs text-gray-600">
          Selected: {value.slice(0, 8).join(", ")}
          {value.length > 8 ? "…" : ""}
        </div>
      ) : (
        <div className="mt-2 text-xs text-gray-400">Selected: none</div>
      )}
    </div>
  );
}

/* ---------- Helpers (Logic) ---------- */

function deriveColumns(payload) {
  const out = new Set();
  if (Array.isArray(payload.columns))
    payload.columns.forEach((c) => out.add(String(c)));
  if (payload.column_metadata && typeof payload.column_metadata === "object") {
    Object.keys(payload.column_metadata).forEach((c) => out.add(String(c)));
  }
  const rows = Array.isArray(payload.preview_data) ? payload.preview_data : [];
  rows.slice(0, 50).forEach((row) => {
    if (row && typeof row === "object")
      Object.keys(row).forEach((k) => out.add(String(k)));
  });
  if (Array.isArray(payload.headers))
    payload.headers.forEach((h) => out.add(String(h)));
  if (payload.schema?.fields && Array.isArray(payload.schema.fields)) {
    payload.schema.fields.forEach((f) => f?.name && out.add(String(f.name)));
  }
  return Array.from(out);
}

function coach(before, after, ops) {
  const msgs = [];
  if (!before || !after) return msgs;

  if (before.shape && after.shape && before.shape[0] !== after.shape[0]) {
    msgs.push(`Rows: ${before.shape[0]} → ${after.shape[0]}`);
  }

  const nb = before.null_counts || {};
  const na = after.null_counts || {};
  const improved = Object.keys(na).filter(
    (k) => (nb[k] ?? 0) > 0 && (na[k] ?? 0) < (nb[k] ?? 0)
  );
  if (improved.length) {
    msgs.push(
      `Nulls reduced in: ${improved.slice(0, 6).join(", ")}${
        improved.length > 6 ? "…" : ""
      }`
    );
  }

  if (ops.lowercase_headers) msgs.push("Lowercased headers.");
  if (ops.remove_duplicates) msgs.push("Removed duplicate rows.");
  if (ops.dropna) msgs.push("Dropped rows with missing values.");
  if (ops.fillna_strategy && (ops.selected_columns?.fillna?.length || 0)) {
    msgs.push(
      `Filled NA (${
        ops.fillna_strategy
      }) in: ${ops.selected_columns.fillna.join(", ")}`
    );
  }
  if (ops.encoding && (ops.selected_columns?.encoding?.length || 0)) {
    msgs.push(
      `Encoded (${ops.encoding}) in: ${ops.selected_columns.encoding.join(
        ", "
      )}`
    );
  }
  if (ops.scale && (ops.selected_columns?.scale?.length || 0)) {
    msgs.push(
      `Scaled (${ops.scale}) in: ${ops.selected_columns.scale.join(", ")}`
    );
  }
  const convPairs = Object.entries(ops.conversions || {});
  if (convPairs.length) {
    msgs.push(
      `Converted: ${convPairs.map(([c, t]) => `${c}→${t}`).join(", ")}`
    );
  }
  const bins = Object.entries(ops.binning || {});
  if (bins.length) {
    msgs.push(`Binned: ${bins.map(([c, b]) => `${c}(${b})`).join(", ")}`);
  }
  return msgs;
}

function normalizeOpsForServer(opts) {
  if (!opts?.lowercase_headers) return opts;
  const lower = (s) => (typeof s === "string" ? s.toLowerCase() : s);
  const clone = JSON.parse(JSON.stringify(opts));
  ["fillna", "scale", "encoding", "outliers"].forEach((k) => {
    clone.selected_columns[k] = (clone.selected_columns[k] || []).map(lower);
  });
  const conv = {};
  Object.entries(clone.conversions || {}).forEach(
    ([k, v]) => (conv[lower(k)] = v)
  );
  clone.conversions = conv;
  const bin = {};
  Object.entries(clone.binning || {}).forEach(([k, v]) => (bin[lower(k)] = v));
  clone.binning = bin;
  return clone;
}

function normalizePipelineForServer(pl, opts) {
  if (!opts?.lowercase_headers) return pl;
  const lower = (s) => (typeof s === "string" ? s.toLowerCase() : s);
  return pl.map((step) => {
    const s = { ...step };
    if (Array.isArray(s.columns)) {
      if (s.op === "conversions" || s.op === "binning") {
        s.columns = s.columns.map((c) =>
          c && typeof c === "object" ? { ...c, name: lower(c.name) } : c
        );
      } else {
        s.columns = s.columns.map((c) => lower(c));
      }
    }
    return s;
  });
}

function summarizePipeline(pl) {
  if (!pl?.length) return "No operations selected.";
  return pl
    .map((s, i) => {
      switch (s.op) {
        case "lowercase_headers":
          return `${i + 1}. Lowercase headers`;
        case "remove_duplicates":
          return `${i + 1}. Remove duplicates`;
        case "conversions":
          return `${i + 1}. Convert dtypes → ${s.columns
            .map((c) => `${c.name}→${c.to}`)
            .join(", ")}`;
        case "binning":
          return `${i + 1}. Bin → ${s.columns
            .map((c) => `${c.name}:${c.bins}`)
            .join(", ")}`;
        case "fillna":
          return `${i + 1}. Fill NA (${s.strategy}) → ${s.columns.join(", ")}`;
        case "dropna":
          return `${i + 1}. Drop NA rows`;
        case "encoding":
          return `${i + 1}. Encode (${s.method}) → ${s.columns.join(", ")}`;
        case "outliers":
          return `${i + 1}. Outliers (${s.method}) → ${s.columns.join(", ")}`;
        case "scale":
          return `${i + 1}. Scale (${s.method}) → ${s.columns.join(", ")}`;
        default:
          return `${i + 1}. ${s.op}`;
      }
    })
    .join("\n");
}

function computeStatsDelta(before, after) {
  if (!before || !after)
    return { summary: "No stats available.", changedNulls: [] };
  const [br, bc] = before.shape || [null, null];
  const [ar, ac] = after.shape || [null, null];
  const shapeDelta =
    br != null && ar != null && bc != null && ac != null
      ? `Shape: ${br}×${bc} → ${ar}×${ac} (${ar - br >= 0 ? "+" : ""}${
          ar - br
        } rows)`
      : "Shape: unknown";

  const nullBefore = before.null_counts || {};
  const nullAfter = after.null_counts || {};
  const changedNulls = [];
  const allCols = new Set([
    ...Object.keys(nullBefore),
    ...Object.keys(nullAfter),
  ]);
  for (const col of allCols) {
    const b = nullBefore[col] ?? 0;
    const a = nullAfter[col] ?? 0;
    if (a !== b) changedNulls.push(`${col}: ${b}→${a}`);
  }
  const nullDelta = changedNulls.length
    ? `Nulls changed: ${changedNulls.slice(0, 8).join(", ")}${
        changedNulls.length > 8 ? "…" : ""
      }`
    : "Nulls unchanged";

  return { summary: `${shapeDelta}\n${nullDelta}`, changedNulls };
}

/* ---------- Main Component ---------- */

export default function DataCleaning() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [previewData, setPreviewData] = useState([]);
  const [cleanedData, setCleanedData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [columnMetadata, setColumnMetadata] = useState(null);
  const [nRows, setNRows] = useState(0);
  const [beforeStats, setBeforeStats] = useState(null);
  const [afterStats, setAfterStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState("");
  const [coachMsgs, setCoachMsgs] = useState([]);
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [visImage, setVisImage] = useState(null);
  const [lastPipelineSummary, setLastPipelineSummary] = useState(""); // Track last sent summary

  const backendUrl =
    process.env.NODE_ENV === "development"
      ? "http://127.0.0.1:8000"
      : process.env.NORTHFLANK_GPU_URL || "";

  // Initialize options with defaults
  const [options, setOptions] = useState({
    fillna_strategy: "",
    scale: "",
    encoding: "",
    lowercase_headers: false,
    dropna: false,
    remove_duplicates: false,
    outlier_method: "",
    conversions: {},
    binning: {},
    selected_columns: { fillna: [], scale: [], encoding: [], outliers: [] },
  });

  // Debounced fetch for backend state updates
  const debouncedFetch = useMemo(
    () =>
      debounce((url, options, setAlerts) => {
        fetch(url, options)
          .then((res) => {
            if (!res.ok)
              throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            return res.json();
          })
          .catch((err) => {
            console.error("Failed to push databot state:", err.message);
            setAlerts((prev) => [
              ...new Set([
                ...prev,
                `Failed to update Databot state: ${err.message}`,
              ]),
            ]);
          });
      }, 500),
    []
  );

  const logAction = (action, extra = {}) => {
    if (!id || isNaN(id)) return;
    fetch(`/api/databot/track/${id}`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    }).catch((err) => {
      setAlerts((prev) => [
        ...new Set([...prev, `Failed to log action: ${err.message}`]),
      ]);
    });
  };

  const pushDatabotState = (payload = {}) => {
    if (!id || isNaN(id)) {
      console.warn("Invalid dataset ID, skipping pushDatabotState");
      return;
    }

    const {
      intent,
      pipeline,
      summary,
      result_summary,
      before_stats,
      after_stats,
      alerts,
    } = payload;
    const plan = summary || (pipeline ? summarizePipeline(pipeline) : "");
    const initial_message = pipeline?.length
      ? "I see you've made some changes to preview on your dataset, would you like me to explain them to you?"
      : null;

    const assistant_hints = [
      intent ? `Intent: ${intent}` : null,
      plan ? `Plan:\n${plan}` : null,
      result_summary ? `Result:\n${result_summary}` : null,
      options.lowercase_headers ? "Lowercase headers enabled." : null,
      options.remove_duplicates ? "Remove duplicates enabled." : null,
      options.dropna ? "Drop NA rows enabled." : null,
      options.fillna_strategy && options.selected_columns?.fillna?.length
        ? `FillNA (${
            options.fillna_strategy
          }) → ${options.selected_columns.fillna.join(", ")}`
        : null,
      options.encoding && options.selected_columns?.encoding?.length
        ? `Encode (${
            options.encoding
          }) → ${options.selected_columns.encoding.join(", ")}`
        : null,
      options.scale && options.selected_columns?.scale?.length
        ? `Scale (${options.scale}) → ${options.selected_columns.scale.join(
            ", "
          )}`
        : null,
      ...Object.entries(options.conversions || {})
        .filter(([, t]) => !!t)
        .map(([c, t]) => `Convert ${c} → ${t}`),
      ...Object.entries(options.binning || {})
        .filter(([, b]) => Number.isFinite(Number(b)) && Number(b) > 1)
        .map(([c, b]) => `Bin ${c} → ${b} bins`),
      "Keep answers concise, <100 words, focusing on dataset changes.", // Encourage brevity
    ].filter(Boolean);

    const body = {
      dataset_id: Number(id),
      options: {
        fillna_strategy: options.fillna_strategy,
        scale: options.scale,
        encoding: options.encoding,
        lowercase_headers: options.lowercase_headers,
        dropna: options.dropna,
        remove_duplicates: options.remove_duplicates,
        outlier_method: options.outlier_method,
        conversions: options.conversions,
        binning: options.binning,
        selected_columns: options.selected_columns,
        assistant_hints,
        intent: intent || undefined,
        last_summary: plan || undefined,
        last_result: result_summary || undefined,
        page: "data-cleaning",
      },
      before_stats,
      after_stats,
      alerts,
    };

    // Only send if the pipeline summary has changed
    if (
      plan !== lastPipelineSummary ||
      intent === "preview" ||
      intent === "preview_result" ||
      intent === "save_request" ||
      intent === "save_success" ||
      intent === "save_error"
    ) {
      setLastPipelineSummary(plan);
      debouncedFetch(
        `${backendUrl}/api/databot/state/${id}`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
        setAlerts
      );

      window.dispatchEvent(
        new CustomEvent("databot:context", {
          detail: {
            datasetId: Number(id),
            page: "data-cleaning",
            intent,
            pipeline,
            summary: plan,
            result_summary,
            before_stats,
            after_stats,
            alerts,
            selectedOpsSummary: summarizePipeline(pipeline),
            initial_message, // Custom initial message
          },
        })
      );
    }
  };

  const pipeline = useMemo(() => {
    const steps = [];

    if (options.lowercase_headers) steps.push({ op: "lowercase_headers" });
    if (options.remove_duplicates) steps.push({ op: "remove_duplicates" });

    const convPairs = Object.entries(options.conversions || {}).filter(
      ([, to]) => !!to
    );
    if (convPairs.length) {
      steps.push({
        op: "conversions",
        columns: convPairs.map(([name, to]) => ({ name, to })),
      });
    }

    const binPairs = Object.entries(options.binning || {}).filter(
      ([, bins]) => Number.isFinite(Number(bins)) && Number(bins) > 1
    );
    if (binPairs.length) {
      steps.push({
        op: "binning",
        columns: binPairs.map(([name, bins]) => ({ name, bins: Number(bins) })),
      });
    }

    if (options.fillna_strategy && options.selected_columns?.fillna?.length) {
      steps.push({
        op: "fillna",
        strategy: options.fillna_strategy,
        columns: options.selected_columns.fillna,
      });
    }

    if (options.dropna) steps.push({ op: "dropna", how: "any" });

    if (options.encoding && options.selected_columns?.encoding?.length) {
      steps.push({
        op: "encoding",
        method: options.encoding,
        columns: options.selected_columns.encoding,
      });
    }

    if (options.outlier_method && options.selected_columns?.outliers?.length) {
      steps.push({
        op: "outliers",
        method: options.outlier_method,
        columns: options.selected_columns.outliers,
      });
    }

    if (options.scale && options.selected_columns?.scale?.length) {
      steps.push({
        op: "scale",
        method: options.scale,
        columns: options.selected_columns.scale,
      });
    }

    const orderIndex = Object.fromEntries(STEP_ORDER.map((k, i) => [k, i]));
    steps.sort((a, b) => (orderIndex[a.op] ?? 999) - (orderIndex[b.op] ?? 999));
    return steps;
  }, [
    options.lowercase_headers,
    options.remove_duplicates,
    options.conversions,
    options.binning,
    options.fillna_strategy,
    options.selected_columns?.fillna,
    options.selected_columns?.encoding,
    options.selected_columns?.outliers,
    options.selected_columns?.scale,
    options.dropna,
    options.encoding,
    options.outlier_method,
    options.scale,
  ]);

  const selectedOpsSummary = useMemo(
    () => summarizePipeline(pipeline),
    [pipeline]
  );

  /* ---------- Load Dataset and Suggestions ---------- */

  useEffect(() => {
    const fetchDataset = async () => {
      if (!id || isNaN(id)) {
        setError("Invalid dataset ID");
        setLoading(false);
        return;
      }
      try {
        const res = await fetch(`/api/datasets/${id}`, {
          credentials: "include",
        });
        if (res.status === 401) return navigate("/login");
        if (res.status === 404) return navigate("/datasets");
        if (!res.ok) throw new Error(`HTTP ${res.status}: Cannot load dataset`);

        const data = await res.json();
        setPreviewData(data.preview_data || []);
        setColumnMetadata(data.column_metadata || null);
        setNRows(data.n_rows || 0);
        setFilename(data.filename || `dataset_${id}`);
        setColumns(deriveColumns(data));
      } catch (err) {
        setError(err.message || "Failed to load dataset");
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [id, navigate]);

  useEffect(() => {
    if (!columns.length) return;
    setOptions((prev) => {
      const keep = new Set(columns);
      const selected = { ...prev.selected_columns };
      ["fillna", "scale", "encoding", "outliers"].forEach((k) => {
        selected[k] = (selected[k] || []).filter((c) => keep.has(c));
      });
      const conv = {};
      Object.entries(prev.conversions || {}).forEach(([k, v]) => {
        if (keep.has(k)) conv[k] = v;
      });
      const bin = {};
      Object.entries(prev.binning || {}).forEach(([k, v]) => {
        if (keep.has(k)) bin[k] = v;
      });
      return {
        ...prev,
        selected_columns: selected,
        conversions: conv,
        binning: bin,
      };
    });
  }, [columns]);

  useEffect(() => {
    const fetchSuggestions = async () => {
      if (!id || isNaN(id)) return;
      try {
        const res = await fetch(
          `${backendUrl}/api/databot/suggestions/${id}?page=data-cleaning`,
          {
            credentials: "include",
          }
        );
        if (res.status === 404) {
          setAlerts((prev) => [
            ...new Set([
              ...prev,
              "Databot suggestions not available for this dataset.",
            ]),
          ]);
          return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        const data = await res.json();
        setAlerts((prev) => [
          ...new Set([...prev, ...(data.suggestions || [])]),
        ]);
      } catch (err) {
        setAlerts((prev) => [
          ...new Set([
            ...prev,
            `Failed to fetch Databot suggestions: ${err.message}`,
          ]),
        ]);
      }
    };
    fetchSuggestions();
  }, [id, backendUrl]);

  // Only push state when pipeline changes meaningfully
  useEffect(() => {
    if (selectedOpsSummary !== lastPipelineSummary && pipeline.length) {
      pushDatabotState({
        intent: "options_updated",
        pipeline,
        summary: selectedOpsSummary,
      });
    }
  }, [pipeline, selectedOpsSummary]);

  /* ---------- Actions ---------- */

  const handlePreview = async () => {
    if (!id || isNaN(id)) {
      setError("Invalid dataset ID");
      return;
    }
    setLoading(true);
    setVisImage(null);
    setError(null);
    setAlerts([]);

    const pl = pipeline;
    const opsForServer = normalizeOpsForServer(options);
    const plForServer = normalizePipelineForServer(pl, options);

    pushDatabotState({
      intent: "preview",
      pipeline: pl,
      summary: summarizePipeline(pl),
    });

    try {
      const dataList = [
        ...(opsForServer.selected_columns?.fillna || []).map((col) => ({
          column: col,
          operation: "fillna",
          value: opsForServer.fillna_strategy,
        })),
        ...Object.entries(opsForServer.conversions || {}).map(
          ([col, type]) => ({
            column: col,
            operation: "convert",
            value: type,
          })
        ),
        ...(opsForServer.selected_columns?.scale || []).map((col) => ({
          column: col,
          operation: "scale",
          value: opsForServer.scale,
        })),
        ...(opsForServer.selected_columns?.encoding || []).map((col) => ({
          column: col,
          operation: "encoding",
          value: opsForServer.encoding,
        })),
        ...(opsForServer.selected_columns?.outliers || []).map((col) => ({
          column: col,
          operation: "outliers",
          value: opsForServer.outlier_method,
        })),
        ...Object.entries(opsForServer.binning || {}).map(([col, bins]) => ({
          column: col,
          operation: "binning",
          value: Number(bins),
        })),
      ].filter((op) => op.value);

      const res = await fetch(`/api/datasets/${id}/clean-preview`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: Number(id),
          save: false,
          pipeline: plForServer,
          operations: opsForServer,
          data: dataList,
          apply_in_order: true,
        }),
      });

      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error((await res.text()) || "Preview failed");

      const resp = await res.json();
      setBeforeStats(resp.before_stats);
      setAfterStats(resp.after_stats);
      setAlerts((prev) => [...new Set([...prev, ...(resp.alerts || [])])]);
      setCleanedData(resp.preview || previewData.slice(0, 10));
      setVisImage(resp.vis_image_base64 || null);

      let action = "Applied cleaning action";
      if (dataList.length > 0) {
        const op = dataList[0];
        if (op.operation === "convert") {
          action = `Transformed datatype for column '${op.column}' to ${op.value}`;
        } else if (op.operation === "fillna") {
          action = `Filled missing values in '${op.column}' with ${op.value}`;
        } else if (op.operation === "scale") {
          action = `Scaled column '${op.column}' using ${op.value}`;
        } else if (op.operation === "encoding") {
          action = `Encoded column '${op.column}' using ${op.value}`;
        } else if (op.operation === "outliers") {
          action = `Removed outliers in '${op.column}' using ${op.value}`;
        } else if (op.operation === "binning") {
          action = `Binned column '${op.column}' into ${op.value} bins`;
        }
      }
      window.dispatchEvent(
        new CustomEvent("dfjsx-cleaning-action", { detail: { action } })
      );

      const delta = computeStatsDelta(resp.before_stats, resp.after_stats);
      setCoachMsgs(coach(resp.before_stats, resp.after_stats, opsForServer));
      pushDatabotState({
        intent: "preview_result",
        pipeline: pl,
        result_summary: delta.summary,
        before_stats: resp.before_stats,
        after_stats: resp.after_stats,
        alerts: resp.alerts || [],
      });

      logAction("preview_success", { steps: pl.length, delta: delta.summary });

      try {
        const sres = await fetch(
          `${backendUrl}/api/databot/suggestions/${id}?page=data-cleaning`,
          {
            credentials: "include",
          }
        );
        if (sres.ok) {
          const sdata = await sres.json();
          setAlerts((prev) => [
            ...new Set([...prev, ...(sdata.suggestions || [])]),
          ]);
        }
      } catch {
        /* non-blocking */
      }
    } catch (err) {
      setError(err.message || "Preview failed");
      setAlerts((prev) => [...new Set([...prev, `Failed: ${err.message}`])]);
      logAction("preview_error", { error: String(err?.message || err) });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!id || isNaN(id)) {
      setError("Invalid dataset ID");
      return;
    }
    setSaving(true);
    setError(null);
    setAlerts([]);

    const opsForServer = normalizeOpsForServer(options);
    const plForServer = normalizePipelineForServer(pipeline, options);

    pushDatabotState({
      intent: "save_request",
      pipeline,
      summary: selectedOpsSummary,
    });

    try {
      const dataList = [
        ...(opsForServer.selected_columns?.fillna || []).map((col) => ({
          column: col,
          operation: "fillna",
          value: opsForServer.fillna_strategy,
        })),
        ...Object.entries(opsForServer.conversions || {}).map(
          ([col, type]) => ({
            column: col,
            operation: "convert",
            value: type,
          })
        ),
        ...(opsForServer.selected_columns?.scale || []).map((col) => ({
          column: col,
          operation: "scale",
          value: opsForServer.scale,
        })),
        ...(opsForServer.selected_columns?.encoding || []).map((col) => ({
          column: col,
          operation: "encoding",
          value: opsForServer.encoding,
        })),
        ...(opsForServer.selected_columns?.outliers || []).map((col) => ({
          column: col,
          operation: "outliers",
          value: opsForServer.outlier_method,
        })),
        ...Object.entries(opsForServer.binning || {}).map(([col, bins]) => ({
          column: col,
          operation: "binning",
          value: Number(bins),
        })),
      ].filter((op) => op.value);

      const res = await fetch(`/api/datasets/${id}/clean-preview`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: Number(id),
          save: true,
          pipeline: plForServer,
          operations: opsForServer,
          data: dataList,
          apply_in_order: true,
        }),
      });

      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error((await res.text()) || "Save failed");

      const data = await res.json();
      setAlerts(data.alerts || []);
      setBeforeStats(data.before_stats);
      setAfterStats(data.after_stats);
      setCleanedData(data.preview || cleanedData);
      setVisImage(data.vis_image_base64 || null);
      setCoachMsgs((prev) => [...prev, "Saved cleaned dataset."]);
      pushDatabotState({
        intent: "save_success",
        pipeline,
        result_summary: "Saved cleaned dataset.",
      });

      if (data.saved) {
        navigate(`/datasets/${id}`);
      } else {
        setAlerts((prev) => [
          ...prev,
          "Save completed but server did not return saved=true.",
        ]);
      }
    } catch (err) {
      setError(err.message || "Save failed");
      setAlerts((prev) => [...prev, `Failed: ${err.message}`]);
      pushDatabotState({
        intent: "save_error",
        pipeline,
        message: err.message || String(err),
      });
    } finally {
      setSaving(false);
    }
  };

  /* ---------- Render Helpers ---------- */

  const renderStats = (stats, title) => (
    <div className="bg-gray-100 p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      {stats?.shape && (
        <p className="text-sm">
          <strong>Rows × Columns:</strong> {stats.shape.join(" × ")}
        </p>
      )}
      {stats?.null_counts && (
        <div className="mt-2">
          <p className="font-medium text-sm">Null Counts:</p>
          <div className="max-h-40 overflow-y-auto border rounded">
            <table className="w-full text-xs">
              <tbody>
                {Object.entries(stats.null_counts).map(([key, val]) => (
                  <tr key={key}>
                    <td className="border px-2 py-1">{key}</td>
                    <td className="border px-2 py-1">{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {stats?.dtypes && (
        <div className="mt-2">
          <p className="font-medium text-sm">Data Types:</p>
          <div className="max-h-40 overflow-y-auto border rounded">
            <table className="w-full text-xs">
              <tbody>
                {Object.entries(stats.dtypes).map(([key, val]) => (
                  <tr key={key}>
                    <td className="border px-2 py-1">{key}</td>
                    <td className="border px-2 py-1">{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );

  const renderPreviewTable = (data, title) => (
    <div className="mt-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div className="overflow-x-auto bg-gray-100 p-4 rounded-lg shadow">
        <table className="w-full text-xs">
          <thead>
            <tr>
              {data.length > 0 &&
                Object.keys(data[0]).map((col) => (
                  <th key={col} className="border px-2 py-1 font-medium">
                    {col}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 5).map((row, i) => (
              <tr key={i} className="hover:bg-gray-200">
                {Object.values(row).map((val, j) => (
                  <td key={j} className="border px-2 py-1">
                    {val === null || val === undefined ? "N/A" : String(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 10 && (
          <p className="mt-2 text-xs text-gray-600">Showing first 5 rows.</p>
        )}
      </div>
    </div>
  );

  const getInfoOutput = () => {
    if (
      !columnMetadata ||
      !Object.keys(columnMetadata).length ||
      !columns.length
    ) {
      return "No dataset information available";
    }
    const dtypeCounts = Object.values(columnMetadata).reduce((acc, meta) => {
      const dtype = meta.dtype || "unknown";
      acc[dtype] = (acc[dtype] || 0) + 1;
      return acc;
    }, {});
    return `<class 'pandas.core.frame.DataFrame'>
RangeIndex: ${nRows} entries, 0 to ${nRows - 1}
Data columns (total ${columns.length} columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
${columns
  .map((name, index) => {
    const meta = columnMetadata[name] || {};
    const nonNullCount =
      meta.null_count !== undefined ? nRows - meta.null_count : "Unknown";
    const dtype = meta.dtype || "unknown";
    return ` ${index.toString().padStart(2, " ")}  ${name.padEnd(30)} ${String(
      nonNullCount
    ).padEnd(15)} ${dtype}`;
  })
  .join("\n")}
dtypes: ${Object.entries(dtypeCounts)
      .map(([dtype, count]) => `${dtype}(${count})`)
      .join(", ")}
memory usage: Unknown`;
  };

  /* ---------- UI ---------- */

  if (loading)
    return (
      <div className="p-4 text-center text-lg text-gray-600">Loading...</div>
    );
  if (error)
    return <div className="p-4 text-center text-lg text-red-500">{error}</div>;

  return (
    <div className="max-w-6xl mx-auto p-4 bg-white rounded-lg shadow space-y-4">
      <h2 className="text-2xl font-bold mb-2 text-gray-800 flex items-center">
        <svg
          className="w-6 h-6 mr-2 text-blue-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-6l-2-2H5a2 2 0 0 0-2 2z"
          />
        </svg>
        Clean Dataset: {filename}
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Section title="Dataset Peek" defaultOpen={false}>
          <details className="text-xs bg-white border border-gray-300 rounded">
            <summary className="p-2 cursor-pointer bg-gray-100 hover:bg-gray-200">
              Dataset Info
            </summary>
            <pre className="p-2 whitespace-pre-wrap">{getInfoOutput()}</pre>
          </details>
        </Section>

        <Section title="Alerts & Suggestions" defaultOpen={true}>
          <select
            className="w-full rounded border-gray-300 px-3 py-2 text-sm bg-yellow-50 text-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-500"
            title="Databot suggestions"
            value={selectedSuggestion}
            onChange={(e) => setSelectedSuggestion(e.target.value)}
          >
            <option value="">Select a suggestion</option>
            {alerts.map((alert, index) => (
              <option key={index} value={alert}>
                {alert}
              </option>
            ))}
          </select>
        </Section>
      </div>

      <Section title="Quick Toggles" defaultOpen={true}>
        <div className="flex items-center flex-wrap gap-4">
          <label className="flex items-center text-sm">
            <input
              type="checkbox"
              checked={options.lowercase_headers}
              onChange={(e) =>
                setOptions((prev) => ({
                  ...prev,
                  lowercase_headers: e.target.checked,
                }))
              }
              className="mr-2"
            />
            Lowercase Headers
          </label>
          <label className="flex items-center text-sm">
            <input
              type="checkbox"
              checked={options.remove_duplicates}
              onChange={(e) =>
                setOptions((prev) => ({
                  ...prev,
                  remove_duplicates: e.target.checked,
                }))
              }
              className="mr-2"
            />
            Remove Duplicates
          </label>
          <label className="flex items-center text-sm">
            <input
              type="checkbox"
              checked={options.dropna}
              onChange={(e) =>
                setOptions((prev) => ({ ...prev, dropna: e.target.checked }))
              }
              className="mr-2"
            />
            Drop NA Rows
          </label>
        </div>
      </Section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Section
          title="Missing Values"
          subtitle="Choose strategy + columns"
          defaultOpen={true}
        >
          <div className="mb-2">
            <select
              onChange={(e) =>
                setOptions((p) => ({ ...p, fillna_strategy: e.target.value }))
              }
              value={options.fillna_strategy}
              className="w-full rounded border-gray-300 px-3 py-1"
              title="How to fill missing values"
            >
              <option value="">No imputation</option>
              <option value="mean">Mean (numeric)</option>
              <option value="median">Median (numeric)</option>
              <option value="mode">Mode (categorical)</option>
              <option value="zero">Zero</option>
              <option value="knn">KNN (numeric)</option>
            </select>
          </div>
          <ColumnPicker
            columns={columns}
            value={options.selected_columns.fillna}
            onChange={(vals) =>
              setOptions((p) => ({
                ...p,
                selected_columns: { ...p.selected_columns, fillna: vals },
              }))
            }
          />
        </Section>

        <Section
          title="Scale numeric columns"
          subtitle="Method + columns"
          defaultOpen={false}
        >
          <div className="mb-2">
            <select
              onChange={(e) =>
                setOptions((p) => ({ ...p, scale: e.target.value }))
              }
              value={options.scale}
              className="w-full rounded border-gray-300 px-3 py-1"
              title="Scale numeric columns"
            >
              <option value="">No scaling</option>
              <option value="normalize">Min-Max (0-1)</option>
              <option value="standardize">Z-Score</option>
              <option value="robust">Robust (IQR)</option>
            </select>
          </div>
          <ColumnPicker
            columns={columns}
            value={options.selected_columns.scale}
            onChange={(vals) =>
              setOptions((p) => ({
                ...p,
                selected_columns: { ...p.selected_columns, scale: vals },
              }))
            }
          />
        </Section>

        <Section
          title="Encode categorical"
          subtitle="Method + columns"
          defaultOpen={false}
        >
          <div className="mb-2">
            <select
              onChange={(e) =>
                setOptions((p) => ({ ...p, encoding: e.target.value }))
              }
              value={options.encoding}
              className="w-full rounded border-gray-300 px-3 py-1"
              title="Encode categorical variables"
            >
              <option value="">No encoding</option>
              <option value="onehot">One-Hot</option>
              <option value="label">Label</option>
              <option value="ordinal">Ordinal</option>
            </select>
          </div>
          <ColumnPicker
            columns={columns}
            value={options.selected_columns.encoding}
            onChange={(vals) =>
              setOptions((p) => ({
                ...p,
                selected_columns: { ...p.selected_columns, encoding: vals },
              }))
            }
          />
        </Section>

        <Section
          title="Handle outliers"
          subtitle="Method + columns"
          defaultOpen={false}
        >
          <div className="mb-2">
            <select
              onChange={(e) =>
                setOptions((p) => ({ ...p, outlier_method: e.target.value }))
              }
              value={options.outlier_method}
              className="w-full rounded border-gray-300 px-3 py-1"
              title="Outliers handling"
            >
              <option value="">No handling</option>
              <option value="iqr">IQR Removal</option>
              <option value="zscore">Z-Score Removal</option>
              <option value="cap">Cap at Percentiles</option>
            </select>
          </div>
          <ColumnPicker
            columns={columns}
            value={options.selected_columns.outliers}
            onChange={(vals) =>
              setOptions((p) => ({
                ...p,
                selected_columns: { ...p.selected_columns, outliers: vals },
              }))
            }
          />
        </Section>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <Section title="Convert data types" defaultOpen={false}>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
            {columns.map((col) => (
              <div key={col} className="flex items-center gap-2">
                <label className="text-xs font-mono w-40 truncate" title={col}>
                  {col}
                </label>
                <select
                  onChange={(e) =>
                    setOptions((p) => ({
                      ...p,
                      conversions: {
                        ...p.conversions,
                        [col]: e.target.value || undefined,
                      },
                    }))
                  }
                  value={options.conversions[col] || ""}
                  className="w-full rounded border-gray-300 px-2 py-1 text-xs"
                  title={`Change dtype of '${col}'`}
                >
                  <option value="">No change</option>
                  <option value="numeric">Numeric</option>
                  <option value="date">Date</option>
                  <option value="category">Category</option>
                </select>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Bin numeric columns" defaultOpen={false}>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
            {columns.map((col) => (
              <div key={col} className="flex items-center gap-2">
                <label className="text-xs font-mono w-40 truncate" title={col}>
                  {col}
                </label>
                <input
                  type="number"
                  min="2"
                  placeholder="Bins"
                  value={options.binning[col] || ""}
                  onChange={(e) => {
                    const num = Number.parseInt(e.target.value, 10);
                    setOptions((p) => ({
                      ...p,
                      binning: {
                        ...p.binning,
                        [col]: Number.isNaN(num) ? undefined : num,
                      },
                    }));
                  }}
                  className="w-full rounded border-gray-300 px-2 py-1 text-xs"
                  title={`Group '${col}' into discrete bins`}
                />
              </div>
            ))}
          </div>
        </Section>
      </div>

      <Section
        title="Selected operations (execution order)"
        defaultOpen={false}
      >
        <pre className="text-xs whitespace-pre-wrap">{selectedOpsSummary}</pre>
      </Section>

      <div className="sticky bottom-2 z-10">
        <div className="flex justify-center gap-4 bg-white/90 backdrop-blur rounded-lg p-3 border shadow">
          <button
            onClick={handlePreview}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500 disabled:opacity-50"
            disabled={loading || saving}
          >
            {loading ? "Previewing..." : "Preview Cleaning"}
          </button>
          {beforeStats && afterStats && (
            <button
              onClick={handleSave}
              disabled={
                saving || alerts.some((a) => String(a).includes("Failed"))
              }
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-500 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save Cleaned Dataset"}
            </button>
          )}
        </div>
      </div>

      {/* {(alerts.length > 0 || error) && (
        <div className="max-w-3xl mx-auto">
          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-3 rounded">
            <p className="text-sm font-medium text-yellow-800">Messages</p>
            <ul className="mt-1 text-xs text-yellow-800 list-disc list-inside space-y-1">
              {error && <li className="text-red-600">{String(error)}</li>}
              {alerts.map((a, i) => (
                <li key={i}>{String(a)}</li>
              ))}
            </ul>
          </div>
        </div>
      )} */}

      {coachMsgs.length > 0 && (
        <div className="max-w-3xl mx-auto">
          <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded">
            <p className="text-sm font-medium text-blue-800">Databot Coach</p>
            <ul className="mt-1 text-xs text-blue-900 list-disc list-inside space-y-1">
              {coachMsgs.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {beforeStats && afterStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {renderStats(beforeStats, "Before Cleaning")}
          {renderStats(afterStats, "After Cleaning")}
        </div>
      )}

      {(previewData.length > 0 || cleanedData.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {previewData.length > 0 &&
            renderPreviewTable(previewData, "Original Data")}
          {cleanedData.length > 0 &&
            renderPreviewTable(cleanedData, "Cleaned Data")}
        </div>
      )}

      {visImage && (
        <div className="text-center">
          <h3 className="text-lg font-semibold mb-2">Correlation Heatmap</h3>
          <img
            src={`data:image/png;base64,${visImage}`}
            alt="Correlation Heatmap"
            className="max-w-full rounded shadow mx-auto"
          />
        </div>
      )}

      <DatabotCoach
        before={beforeStats}
        after={afterStats}
        pipeline={pipeline}
        options={options}
      />
    </div>
  );
}
