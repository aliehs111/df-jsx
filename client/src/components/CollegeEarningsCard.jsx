// client/src/wherever/CollegeEarningsCard.jsx
import React, { useEffect, useState } from "react";
import TrainingReportPDF from "../assets/college_earnings_v1_75K_5y.pdf";
export default function CollegeEarningsCard() {
  // --- local state (hooks must be inside the component) ---
  const [cip4, setCip4] = useState("1101");
  const [degree, setDegree] = useState("Bachelor");
  const [state, setState] = useState("CA");
  const [pubPriv, setPubPriv] = useState("");

  const [degreeLevels, setDegreeLevels] = useState([]);
  const [statesList, setStatesList] = useState([]);
  const [cip4List, setCip4List] = useState([]);
  const [pubPrivList, setPubPrivList] = useState([]);

  const [loading, setLoading] = useState(false);
  const [optsLoading, setOptsLoading] = useState(true);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [cip4Options, setCip4Options] = useState([]);

  const API_BASE =
    import.meta.env.MODE === "development" ? "http://127.0.0.1:8000" : "";

  const canPredict = cip4 && degree && state;

  // --- fetch encoders once on mount ---
  useEffect(() => {
    async function loadEncoders() {
      try {
        const res = await fetch(
          `${API_BASE}/api/predictors/college_earnings/v1_75k_5y/encoders`,
          { credentials: "include" }
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const enc = await res.json();

        // keep existing lists (useful for counts badge, etc.)
        setDegreeLevels(enc.degree_levels || []);
        setStatesList(enc.states || []);
        setCip4List(enc.cip4 || []);
        setPubPrivList(enc.public_private || []);

        // NEW: build labeled CIP4 options
        const opts =
          enc.cip4_options ||
          (enc.cip4 || []).map((code) => ({ code, label: code }));
        // optional: sort alphabetically by label
        opts.sort((a, b) => a.label.localeCompare(b.label));
        setCip4Options(opts);

        // set defaults if available
        if ((enc.degree_levels || []).length) setDegree(enc.degree_levels[0]);
        if ((enc.states || []).length)
          setState(enc.states.includes("CA") ? "CA" : enc.states[0]);

        // default CIP selection prefers 1101 if present, else first option
        if (opts.length) {
          const def = opts.some((o) => o.code === "1101")
            ? "1101"
            : opts[0].code;
          setCip4(def);
        }
      } catch (e) {
        console.error("Failed to load encoders", e);
        setError("Couldn’t load dropdown options.");
      } finally {
        setOptsLoading(false);
      }
    }
    loadEncoders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePredict = async () => {
    if (!canPredict || loading) return;
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predictors/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          model: "college_earnings_v1_75k_5y",
          params: {
            cip4,
            degree_level: degree,
            state,
            ...(pubPriv ? { public_private: pubPriv } : {}),
          },
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const openModelBot = () => {
    if (!result) return;
    const ctx = {
      bot: "ModelBot",
      page: "predictors",
      feature: "college_earnings_v1_75k_5y",
      version: result.version || "v1_75k_5y",
      inputs: {
        cip4,
        degree_level: degree,
        state,
        public_private: pubPriv || null,
      },
      result: {
        prob: result.probability,
        bucket: result.risk_bucket,
        drivers: result.drivers || [],
      },
      warnings: result.warnings || [],
    };
    window.dispatchEvent(
      new CustomEvent("dfjsx-open-bot", {
        detail: { botType: "modelbot", context: ctx },
      })
    );
  };
  console.log("CE_CARD_LIVE_MARK", new Date().toISOString());

  // --- UI ---
  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 p-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium">College Earnings — 5y ≥ $75k</h2>

          <p className="text-xs text-gray-500">
            Hierarchical logistic (CIP + State RE)
          </p>
        </div>

        {result?.risk_bucket && (
          <span className="inline-flex items-center rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium">
            {result.risk_bucket}
          </span>
        )}
      </div>

      <div className="text-[10px] text-gray-400 px-4 pt-2">
        encoders: {degreeLevels.length} deg • {statesList.length} states •{" "}
        {pubPrivList.length} types • {cip4List.length} CIP4
      </div>

      <div className="p-4 space-y-4">
        {/* Inputs */}
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-sm font-medium">
              Program (CIP-4)
            </span>
            <select
              className="w-full rounded-xl border border-gray-300 p-2.5 text-sm focus:outline-none focus:ring-2"
              value={cip4}
              onChange={(e) => setCip4(e.target.value)}
              disabled={optsLoading}
            >
              {cip4Options.map((opt) => (
                <option key={opt.code} value={opt.code}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-sm font-medium">Degree level</span>
            <select
              className="w-full rounded-xl border border-gray-300 p-2.5 text-sm focus:outline-none focus:ring-2"
              value={degree}
              onChange={(e) => setDegree(e.target.value)}
              disabled={optsLoading}
            >
              {degreeLevels.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-sm font-medium">State</span>
            <select
              className="w-full rounded-xl border border-gray-300 p-2.5 text-sm focus:outline-none focus:ring-2"
              value={state}
              onChange={(e) => setState(e.target.value)}
              disabled={optsLoading}
            >
              {statesList.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-sm font-medium">
              Institution type (optional)
            </span>
            <select
              className="w-full rounded-xl border border-gray-300 p-2.5 text-sm focus:outline-none focus:ring-2"
              value={pubPriv}
              onChange={(e) => setPubPriv(e.target.value)}
              disabled={optsLoading}
            >
              <option value="">--</option>
              {pubPrivList.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={handlePredict}
            disabled={!canPredict || loading || optsLoading}
            className="inline-flex items-center rounded-2xl bg-primary px-4 py-2 text-white hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              "Predicting..."
            ) : (
              <>
                <LightningIcon className="mr-2 h-4 w-4" />
                Predict
              </>
            )}
          </button>

          {result && (
            <button
              onClick={openModelBot}
              className="btn btn-lg relative overflow-hidden bg-orange-500 text-white font-semibold shadow-md hover:bg-orange-600 hover:scale-105 transition-all duration-200 group"
            >
              <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent translate-x-[-150%] animate-shine group-hover:animate-shine" />
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
              Ask Databot about these results!
            </button>
          )}
          {error && <div className="text-sm text-red-600">{error}</div>}
        </div>

        {/* Result */}
        {result ? (
          <div className="space-y-3">
            <div>
              <div className="text-sm text-gray-500">
                Probability (5y ≥ $75k)
              </div>
              <div className="mt-1 text-3xl font-semibold">
                {result.probability != null
                  ? `${Math.round(result.probability * 100)}%`
                  : "–"}
              </div>
              {result.confidence && (
                <div className="text-xs text-gray-500 mt-1">
                  Confidence: {result.confidence}
                </div>
              )}
            </div>

            {!!result.drivers?.length && (
              <div>
                <div className="mb-2 text-sm font-medium">Top drivers</div>
                <ul className="space-y-2">
                  {result.drivers.map((d, idx) => (
                    <li
                      key={idx}
                      className="rounded-xl border border-gray-200 p-3 text-sm"
                    >
                      <span className="font-semibold">
                        {d.direction === "+" ? "↑" : "↓"} {d.factor}
                      </span>
                      {d.weight != null && (
                        <span className="ml-2 text-gray-500">({d.weight})</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="pt-1">
              <a
                href={TrainingReportPDF}
                target="_blank"
                rel="noreferrer"
                className="text-sm text-blue-600 hover:underline"
              >
                View training report (v1)
              </a>
            </div>
            <div className="pt-1">
              <a
                href="https://collegescorecard.ed.gov/data/"
                target="_blank"
                rel="noreferrer"
                className="text-sm text-blue-600 hover:underline"
              >
                View source data
              </a>
            </div>

            {!!result.warnings?.length && (
              <div className="rounded-xl border border-yellow-300 bg-yellow-50 p-3 text-sm">
                <div className="font-medium">Warnings</div>
                <ul className="list-disc pl-5">
                  {result.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="text-sm text-gray-500">
            Fill fields and click Predict.
          </div>
        )}
      </div>
    </div>
  );
}

function LightningIcon({ className = "" }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path d="M11 21 20 8h-7l2-7L4 10h7l-2 11z" />
    </svg>
  );
}
