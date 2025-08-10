import { useEffect, useMemo, useRef, useState } from "react";

// Advanced Predictors page for df-jsx
// - Adds a Settings panel to adjust model sensitivity, thresholds, category toggles, and audience effect.
// - Sends optional `overrides` with the request (backend can merge with defaults if supported).
// - Reuses global DataBot as "ModelBot" via window event.

const AUDIENCES = [
  { label: "ESL", value: "ESL" },
  { label: "Older Adults", value: "OlderAdults" },
  { label: "General", value: "General" },
];

const MEDIA = [
  { label: "SMS", value: "SMS" },
  { label: "Email", value: "Email" },
];

const INTENTS = [
  { label: "-- Optional --", value: "" },
  { label: "Reminder", value: "Reminder" },
  { label: "Action Required", value: "ActionRequired" },
  { label: "FYI", value: "FYI" },
];

export default function PredictorsPro() {
  // Core inputs
  const [text, setText] = useState("");
  const [audience, setAudience] = useState("General");
  const [medium, setMedium] = useState("Email");
  const [intent, setIntent] = useState("");

  // Network state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // Settings state
  const [showSettings, setShowSettings] = useState(false);
  const [includeOverrides, setIncludeOverrides] = useState(true);
  const [sensitivity, setSensitivity] = useState(1.0); // 0.5 .. 1.5
  const [audienceSoften, setAudienceSoften] = useState(0.6); // 0..1
  const [low, setLow] = useState(0.25);
  const [high, setHigh] = useState(0.65);
  const [enabled, setEnabled] = useState({
    idioms: true,
    jargon: true,
    ambiguous_time: true,
    polysemy: true,
    numeracy_date: true,
  });

  const predictBtnRef = useRef(null);

  const canPredict = useMemo(
    () => text.trim().length > 0 && !!audience && !!medium,
    [text, audience, medium]
  );

  useEffect(() => {
    // Keyboard shortcut: Cmd/Ctrl+Enter to run
    const onKey = (e) => {
      const metaEnter = (e.metaKey || e.ctrlKey) && e.key === "Enter";
      if (metaEnter && canPredict && !loading) {
        e.preventDefault();
        predictBtnRef.current?.click();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canPredict, loading]);

  const handlePredict = async () => {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const payload = {
        model: "accessibility_risk",
        params: {
          text: text.trim(),
          audience,
          medium,
          intent: intent || null,
        },
      };

      if (includeOverrides) {
        payload.overrides = {
          sensitivity: Number(sensitivity),
          audience_soften: Number(audienceSoften),
          density_thresholds: { low: Number(low), high: Number(high) },
          enable_categories: Object.entries(enabled)
            .filter(([_, v]) => v)
            .map(([k]) => k),
          // audience_weights: { ESL: 1.15, OlderAdults: 1.1 } // optional live override example
        };
      }

      const res = await fetch("/api/predictors/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const msg = await safeErr(res);
        throw new Error(msg || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);

      // Build and store context for DataBot (do not open)
      const ctx = {
        bot: "ModelBot",
        feature: "accessibility_risk",
        version: data.version || "v1",
        inputs: {
          text: text.trim().slice(0, 800),
          audience,
          medium,
          intent: intent || null,
        },
        result: {
          prob: data.misinterpretation_probability,
          bucket: data.risk_bucket,
          confusion_sources: data.confusion_sources,
          rewrite: data.rewrite_15_words,
        },
      };

      // Persist for the chat session without opening the panel
      window.dispatchEvent(
        new CustomEvent("dfjsx-set-bot-context", {
          detail: { botType: "modelbot", context: ctx },
        })
      );
    } catch (err) {
      console.error(err);
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const openModelBot = () => {
    if (!result) return;
    const ctx = {
      bot: "ModelBot",
      feature: "accessibility_risk",
      version: result.version || "v1",
      inputs: {
        text: text.trim().slice(0, 800),
        audience,
        medium,
        intent: intent || null,
      },
      result: {
        prob: result.misinterpretation_probability,
        bucket: result.risk_bucket,
        confusion_sources: result.confusion_sources,
        rewrite: result.rewrite_15_words,
      },
    };

    window.dispatchEvent(
      new CustomEvent("dfjsx-open-bot", {
        detail: { botType: "modelbot", context: ctx },
      })
    );
  };

  return (
    <div className="mx-auto max-w-7xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Predictor Models
          </h1>
          <p className="text-sm text-gray-500">
            Accessibility Misinterpretation Risk — adjustable settings
          </p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Left: Inputs & Settings */}
        <div className="rounded-2xl border border-gray-200 bg-white shadow-sm">
          <div className="border-b border-gray-100 p-4 flex items-center justify-between">
            <h2 className="text-lg font-medium">Inputs</h2>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={includeOverrides}
                  onChange={(e) => setIncludeOverrides(e.target.checked)}
                />
                <span>Apply Settings</span>
              </label>
              <button
                onClick={() => setShowSettings((s) => !s)}
                className="rounded-xl border bg-teal-300 px-3 py-1.5 text-sm hover:bg-gray-50"
              >
                {showSettings ? "Hide Settings" : "Show Settings"}
              </button>
            </div>
          </div>

          <div className="p-4 space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">
                Notification text
              </label>
              <textarea
                className="w-full rounded-xl border border-gray-300 p-3 focus:outline-none focus:ring-2"
                rows={8}
                placeholder="Paste the message your users will receive..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="mt-1 text-xs text-gray-400">
                Tip: Cmd/Ctrl+Enter to Predict
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <Select
                label="Audience"
                value={audience}
                onChange={setAudience}
                options={AUDIENCES}
              />
              <Select
                label="Medium"
                value={medium}
                onChange={setMedium}
                options={MEDIA}
              />
              <Select
                label="Intent (optional)"
                value={intent}
                onChange={setIntent}
                options={INTENTS}
              />
            </div>

            {showSettings && (
              <SettingsPanel
                sensitivity={sensitivity}
                setSensitivity={setSensitivity}
                audienceSoften={audienceSoften}
                setAudienceSoften={setAudienceSoften}
                low={low}
                setLow={setLow}
                high={high}
                setHigh={setHigh}
                enabled={enabled}
                setEnabled={setEnabled}
              />
            )}

            <div className="flex items-center gap-3">
              <button
                ref={predictBtnRef}
                onClick={handlePredict}
                disabled={!canPredict || loading}
                className="inline-flex items-center rounded-2xl bg-black px-4 py-2 text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <Spinner label="Predicting" />
                ) : (
                  <>
                    <LightningIcon className="mr-2 h-4 w-4" /> Predict
                  </>
                )}
              </button>
              {error && <div className="text-sm text-red-600">{error}</div>}
            </div>
          </div>
        </div>

        {/* Right: Result */}
        <div className="rounded-2xl border border-gray-200 bg-white shadow-sm">
          <div className="border-b border-gray-100 p-4 flex items-center justify-between">
            <h2 className="text-lg font-medium">Result</h2>
            {result && (
              <BucketChip
                prob={result.misinterpretation_probability}
                bucket={result.risk_bucket}
              />
            )}
          </div>

          <div className="p-4 space-y-5">
            {!result && (
              <div className="text-sm text-gray-500">
                Run a prediction to see results here.
              </div>
            )}

            {result && (
              <>
                <div>
                  <div className="text-sm text-gray-500">
                    Misinterpretation probability
                  </div>
                  <div className="mt-1 text-3xl font-semibold">
                    {formatPct(result.misinterpretation_probability)}
                  </div>
                </div>

                {!!result.confusion_sources?.length && (
                  <div>
                    <div className="mb-2 text-sm font-medium">
                      Top confusion sources
                    </div>
                    <ul className="space-y-2">
                      {result.confusion_sources.map((c, idx) => (
                        <li
                          key={idx}
                          className="rounded-xl border border-gray-200 p-3"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold">
                              {c.type}
                            </span>
                            {c.note && (
                              <span className="text-xs text-gray-500">
                                {c.note}
                              </span>
                            )}
                          </div>
                          {!!c.evidence?.length && (
                            <div className="mt-1 text-sm text-gray-700">
                              Evidence:{" "}
                              <span className="italic">
                                {c.evidence.join(", ")}
                              </span>
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.rewrite_15_words && (
                  <div>
                    <div className="mb-2 text-sm font-medium">
                      ≤15-word rewrite
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex-1 rounded-xl border border-gray-200 bg-gray-50 p-3 text-sm">
                        {result.rewrite_15_words}
                      </div>
                      <button
                        onClick={() => copyText(result.rewrite_15_words)}
                        className="inline-flex items-center rounded-2xl border border-gray-300 px-3 py-2 text-sm hover:bg-gray-50"
                        title="Copy to clipboard"
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                )}

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

                <div className="pt-2">
                  <button
                    onClick={openModelBot}
                    className="inline-flex items-center rounded-2xl border border-gray-300 px-3 py-2 text-sm font-medium shadow-sm hover:bg-gray-50"
                  >
                    Why these results?
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingsPanel({
  sensitivity,
  setSensitivity,
  audienceSoften,
  setAudienceSoften,
  low,
  setLow,
  high,
  setHigh,
  enabled,
  setEnabled,
}) {
  return (
    <div className="rounded-xl border border-gray-200 p-3 space-y-3 bg-gray-50">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          Sensitivity ({Number(sensitivity).toFixed(2)})
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.05"
            value={sensitivity}
            onChange={(e) => setSensitivity(parseFloat(e.target.value))}
            className="w-full"
          />
        </label>
        <label className="block text-sm">
          Audience Soften ({Number(audienceSoften).toFixed(2)})
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={audienceSoften}
            onChange={(e) => setAudienceSoften(parseFloat(e.target.value))}
            className="w-full"
          />
        </label>
        <label className="block text-sm">
          Low threshold
          <input
            className="w-full rounded border px-2 py-1"
            value={low}
            onChange={(e) => setLow(e.target.value)}
            type="number"
            step="0.01"
          />
        </label>
        <label className="block text-sm">
          High threshold
          <input
            className="w-full rounded border px-2 py-1"
            value={high}
            onChange={(e) => setHigh(e.target.value)}
            type="number"
            step="0.01"
          />
        </label>
      </div>

      <div className="text-sm">
        Categories
        <div className="mt-2 flex flex-wrap gap-3">
          {Object.keys(enabled).map((k) => (
            <label
              key={k}
              className="inline-flex items-center gap-2 rounded-xl border px-3 py-1 bg-white"
            >
              <input
                type="checkbox"
                checked={enabled[k]}
                onChange={(e) =>
                  setEnabled({ ...enabled, [k]: e.target.checked })
                }
              />
              <span className="capitalize">{k.replace(/_/g, " ")}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      <select
        className="w-full rounded-xl border border-gray-300 p-2.5 text-sm focus:outline-none focus:ring-2"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function BucketChip({ prob, bucket }) {
  const label = bucket || bucketFromProb(prob);
  return (
    <span className="inline-flex items-center rounded-full border border-gray-200 px-2.5 py-1 text-xs font-medium">
      {label}
    </span>
  );
}

function bucketFromProb(p) {
  if (p == null || Number.isNaN(p)) return "Unknown";
  if (p >= 0.66) return "High";
  if (p >= 0.33) return "Medium";
  return "Low";
}

function formatPct(p) {
  if (p == null || Number.isNaN(p)) return "–";
  const n = Math.max(0, Math.min(100, Math.round(p * 100)));
  return `${n}%`;
}

async function safeErr(res) {
  try {
    const t = await res.text();
    return t || res.statusText;
  } catch {
    return res.statusText;
  }
}

async function copyText(t) {
  try {
    await navigator.clipboard.writeText(t);
  } catch (err) {
    const ta = document.createElement("textarea");
    ta.value = t;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
}

function Spinner({ label }) {
  return (
    <span className="inline-flex items-center">
      <svg
        className="mr-2 h-4 w-4 animate-spin"
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          cx="12"
          cy="12"
          r="10"
          strokeWidth="4"
          stroke="currentColor"
          opacity="0.2"
        />
        <path
          d="M22 12a10 10 0 0 1-10 10"
          strokeWidth="4"
          stroke="currentColor"
        />
      </svg>
      {label}
    </span>
  );
}

function LightningIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={props.className}>
      <path d="M11 21 20 8h-7l2-7L4 10h7l-2 11z" />
    </svg>
  );
}
