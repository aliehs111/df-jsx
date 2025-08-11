import { useState } from "react";

const DEGREE_LEVELS = [
  "Associate",
  "Bachelor",
  "Master",
  "Professional",
  "Doctoral",
];
const STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]; // extend later
const CIP4 = [
  { code: "1101", label: "Computer Science (11.01)" },
  { code: "5203", label: "Accounting (52.03)" },
  { code: "1401", label: "Engineering (14.01)" },
  { code: "2601", label: "Biological Sciences (26.01)" },
]; // swap for your real list later

export default function CollegeEarningsCard() {
  const [cip4, setCip4] = useState("1101");
  const [degree, setDegree] = useState("Bachelor");
  const [state, setState] = useState("CA");
  const [pubPriv, setPubPriv] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const API_BASE =
    import.meta.env.MODE === "development" ? "http://127.0.0.1:8000" : "";

  const canPredict = cip4 && degree && state;

  const handlePredict = async () => {
    if (!canPredict || loading) return;
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/predictors/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  // Optional: open your global ModelBot with this prediction’s context
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
            >
              {CIP4.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.label}
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
            >
              {DEGREE_LEVELS.map((d) => (
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
            >
              {STATES.map((s) => (
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
            >
              <option value="">--</option>
              <option value="Public">Public</option>
              <option value="Private">Private</option>
            </select>
          </label>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={handlePredict}
            disabled={!canPredict || loading}
            className="inline-flex items-center rounded-2xl bg-black px-4 py-2 text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          {result && (
            <button
              onClick={openModelBot}
              className="inline-flex items-center rounded-2xl border border-gray-300 px-3 py-2 text-sm font-medium shadow-sm hover:bg-gray-50"
            >
              Why these results?
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
              {/* When you upload the PDF, swap href to the real S3 URL */}
              <a
                href="#"
                onClick={(e) => e.preventDefault()}
                className="text-sm text-blue-600 hover:underline"
                title="Training report (v1) — upload PDF and set the link"
              >
                View training report (v1)
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
