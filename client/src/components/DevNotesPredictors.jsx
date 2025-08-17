// client/src/components/DevNotesPredictors.jsx
import { useState, Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";
import {
  XMarkIcon,
  InformationCircleIcon,
  ClipboardIcon,
  CheckIcon,
} from "@heroicons/react/24/outline";

/* ---------- Small helper: uniform section block ---------- */

function Section({ title, as, children }) {
  const Title = (
    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
      {title}
    </div>
  );

  if (as === "ul") {
    return (
      <div className="space-y-1">
        {Title}
        <ul className="list-disc list-inside text-gray-700 space-y-1">
          {children}
        </ul>
      </div>
    );
  }

  if (as === "ol") {
    return (
      <div className="space-y-1">
        {Title}
        <ol className="list-decimal list-inside text-gray-700 space-y-1">
          {children}
        </ol>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {Title}
      <div className="text-gray-700">{children}</div>
    </div>
  );
}

/* ---------- Small docs-like code block with copy ---------- */
function CodeBlock({ code = "", language = "text", dark = true }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  };
  return (
    <div className="rounded-lg overflow-hidden border border-gray-200">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50">
        <span className="text-[11px] font-medium tracking-wide text-gray-600 uppercase">
          {language}
        </span>
        <button
          onClick={onCopy}
          className="inline-flex items-center gap-1 text-xs text-gray-600 hover:text-gray-900"
          title="Copy to clipboard"
        >
          {copied ? (
            <CheckIcon className="h-4 w-4" />
          ) : (
            <ClipboardIcon className="h-4 w-4" />
          )}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre
        className={[
          "p-3 overflow-x-auto text-xs leading-relaxed font-mono",
          dark ? "bg-slate-900 text-slate-100" : "bg-slate-50 text-slate-800",
        ].join(" ")}
      >
        {code}
      </pre>
    </div>
  );
}

export default function DevNotesPredictors() {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("access");

  /* ---------- Accessibility predictor snippets ---------- */
  const accessFrontend = `const payload = {
  model: "accessibility_risk",
  params: {
    text,
    audience,
    medium,
    intent,
    thresholds: { low, high },
    sensitivity,
    audience_soften
  }
};

const res = await fetch("/api/predictors/infer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  credentials: "include",
  body: JSON.stringify(payload),
});

const data = await res.json();
setResult(data);`;

  const accessBackend = `# server/predictors/accessibility_risk.py
def run_accessibility_risk(params):
    text = params["text"]
    clean = preprocess_text(text)
    feats = extract_features(clean)         # regex counts, readability, etc.
    prob = clf.predict_proba([feats])[0, 1] # probability of misinterpretation
    return {
        "model": "accessibility_risk",
        "version": "v1",
        "misinterpretation_probability": float(prob),
        "risk_bucket": bucketize(prob, params.get("thresholds")),
        "confusion_sources": detect_confusion_sources(clean),
        "rewrite_15_words": suggest_rewrite(clean)
    }`;

  const accessSample = `{
  "model": "accessibility_risk",
  "version": "v1",
  "misinterpretation_probability": 0.15,
  "risk_bucket": "Low",
  "confusion_sources": [{"type":"dates","evidence":["9/15"]}],
  "rewrite_15_words": "Please pay half the amount by September 15."
}`;

  /* ---------- College Earnings predictor snippets ---------- */
  const earnTrain = `# notebook: train_college_earnings.ipynb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("earnings_5y.csv")
X = df[["cip4","degree_level","state","public_private"]]
y = (df["earn_5y"] >= 75000).astype(int)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
  ["cip4","degree_level","state","public_private"])])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])

Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(Xtr, ytr)
print("Holdout accuracy:", pipe.score(Xte, yte))
joblib.dump(pipe, "artifacts/college_earnings_v1.joblib")`;

  const earnInfer = `# server/predictors/college_earnings.py
import joblib, pandas as pd
from .utils import bucketize_prob, explain_drivers

MODEL_PATH = "artifacts/college_earnings_v1.joblib"
_pipe = joblib.load(MODEL_PATH)

def run_college_earnings(params):
    row = {
        "cip4": params.get("cip4"),
        "degree_level": params.get("degree_level"),
        "state": params.get("state"),
        "public_private": params.get("public_private"),
    }
    X = pd.DataFrame([row])
    prob = float(_pipe.predict_proba(X)[0, 1])
    return {
        "model": "college_earnings",
        "version": "v1",
        "inputs": row,
        "prob": prob,
        "bucket": bucketize_prob(prob),
        "drivers": explain_drivers(_pipe, X)
    }`;

  const earnSample = `{
  "model": "college_earnings",
  "version": "v1",
  "prob": 0.72,
  "bucket": "High",
  "drivers": ["degree_level=Bachelor", "cip4=1101", "state=CA"]
}`;

  return (
    <>
      {/* Floating button — bottom-left (consistent across pages) */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-[60] rounded-full bg-accent p-3 shadow-lg hover:bg-accent/90 text-white"
        title="Predictors — Dev Notes"
        aria-label="Open Dev Notes"
      >
        <InformationCircleIcon className="h-6 w-6" />
      </button>

      {/* Slide-over on the right */}
      <Transition.Root show={open} as={Fragment}>
        <Dialog as="div" className="relative z-[999]" onClose={setOpen}>
          {/* Dim overlay */}
          <Transition.Child
            as={Fragment}
            enter="ease-in-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in-out duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-700/40 transition-opacity" />
          </Transition.Child>

          {/* Right sheet */}
          <div className="fixed inset-0 overflow-hidden">
            <div className="absolute inset-0 overflow-hidden">
              <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
                <Transition.Child
                  as={Fragment}
                  enter="transform transition ease-in-out duration-300 sm:duration-500"
                  enterFrom="translate-x-full"
                  enterTo="translate-x-0"
                  leave="transform transition ease-in-out duration-300 sm:duration-500"
                  leaveFrom="translate-x-0"
                  leaveTo="translate-x-full"
                >
                  <Dialog.Panel className="pointer-events-auto w-screen max-w-md bg-white shadow-2xl flex flex-col">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
                      <div>
                        <Dialog.Title className="text-lg font-semibold text-gray-900">
                          Predictors — Dev Notes
                        </Dialog.Title>
                        {/* Tabs */}
                        <div className="mt-2 flex gap-2">
                          <button
                            onClick={() => setTab("access")}
                            className={`px-2.5 py-1 rounded-md text-xs font-medium ring-1 transition ${
                              tab === "access"
                                ? "bg-indigo-600 text-white ring-indigo-600"
                                : "bg-white text-gray-700 ring-gray-300 hover:bg-gray-50"
                            }`}
                          >
                            Accessibility Predictor
                          </button>
                          <button
                            onClick={() => setTab("earnings")}
                            className={`px-2.5 py-1 rounded-md text-xs font-medium ring-1 transition ${
                              tab === "earnings"
                                ? "bg-indigo-600 text-white ring-indigo-600"
                                : "bg-white text-gray-700 ring-gray-300 hover:bg-gray-50"
                            }`}
                          >
                            College Earnings Predictor
                          </button>
                        </div>
                      </div>

                      <button
                        onClick={() => setOpen(false)}
                        className="text-gray-400 hover:text-gray-600"
                        aria-label="Close Dev Notes"
                      >
                        <XMarkIcon className="h-6 w-6" />
                      </button>
                    </div>

                    {/* Body */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6 text-sm">
                      {tab === "access" && (
                        <>
                          <h3 className="text-base font-semibold text-gray-900">
                            Clarity Predictor (Beta) — Dev Notes
                          </h3>

                          <Section title="How it works">
                            This isn’t a trained ML model yet. I started with
                            some testing json files to get the logic to work. I
                            wanted to train a model but could not find the right
                            data to use. So I came up with the idea that since
                            the open AI chat models area already trained on this
                            topic, I could just slot in a route to complete the
                            response. The backend runs a deterministic rules
                            scorer (idioms, jargon, vague time, polysemy,
                            numbers/dates) to estimate{" "}
                            <strong>misinterpretation risk</strong>, and OpenAI
                            (Databot) generates the rewrite.
                            <div className="mt-2 rounded-md bg-sky-50 border border-sky-200 px-3 py-2 text-sky-900 text-xs">
                              Flow (today): UI →{" "}
                              <code>/api/predictors/infer</code> (rules) → UI
                              calls
                              <code>/api/databot/query</code> (OpenAI) →
                              overwrite <code>rewrite_15_words</code> with the
                              LLM answer.
                            </div>
                          </Section>

                          <Section title="Why the probability skews Low">
                            The score is a weighted count of flags divided by
                            token count, passed through a conservative sigmoid
                            (center≈0.35, k≈6). For most short messages, density
                            is small → <em>base</em> stays low. I kept the
                            probability to show trend/bucketing and for future
                            calibration. Ideally, I want the score to be learned
                            from data (so Medium/High triggers behave more
                            naturally).
                          </Section>

                          <Section title="Future plan (real model)">
                            <ul>
                              <li>
                                Assemble labeled data (plain vs. complex,
                                idioms/jargon/time ambiguity).
                              </li>
                              <li>
                                Train a small classifier (TF-IDF + metadata,
                                Logistic Regression + calibration).
                              </li>
                              <li>
                                Serve via FastAPI behind the same API; keep
                                rule-based explanations.
                              </li>
                              <li>
                                Tune thresholds so buckets reflect reality (not
                                “always Low”).
                              </li>
                            </ul>
                          </Section>

                          <Section title="Inputs">
                            Text, audience, medium, (optional) intent. Optional
                            overrides: sensitivity, thresholds (low/high),
                            audience_soften, and enabled categories.
                          </Section>

                          <Section title="Output">
                            Probability, risk bucket, confusion sources
                            (evidence + notes), and a ≤15-word rewrite
                            (LLM-polished).
                          </Section>

                          <Section title="Snippet — Frontend (actual flow)">
                            <CodeBlock
                              language="javascript"
                              dark
                              code={`// 1) Call the rules-based scorer
const res = await fetch(\`\${API_BASE}/api/predictors/infer\`, {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify({
    model: "accessibility_risk",
    params: { text, audience, medium, intent: intent || null },
    overrides: includeOverrides ? {
      sensitivity: Number(sensitivity),
      audience_soften: Number(audienceSoften),
      density_thresholds: { low: Number(low), high: Number(high) },
      enable_categories: Object.entries(enabled).filter(([,v]) => v).map(([k]) => k)
    } : undefined
  })
});
const data = await res.json();
const localRewrite = data.rewrite_15_words;

// 2) Ask Databot (OpenAI) to polish (overwrites local rewrite)
const databotRes = await fetch(\`\${API_BASE}/api/databot/query\`, {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  credentials: "include",
  body: JSON.stringify({
    question: \`Rewrite the following text to be clear and concise for \${audience} via \${medium}, max 15 words: "\${text.trim()}"\`,
    bot_type: "modelbot",
    model_context: {
      feature: "accessibility_risk",
      inputs: { text: text.trim(), audience, medium, intent: intent || null },
      result: {
        prob: data.misinterpretation_probability,
        bucket: data.risk_bucket,
        confusion_sources: data.confusion_sources,
        rewrite: localRewrite // (optional: seed Databot with local rewrite)
      }
    }
  })
});
const databotData = await databotRes.json();
data.llm_rewrite_15_words = (databotData.answer || "").trim();
data.rewrite_15_words = data.llm_rewrite_15_words || localRewrite; // display LLM, fallback to local
setResult(data);`}
                            />
                          </Section>

                          <Section title="Snippet — Backend (rules core, no ML)">
                            <CodeBlock
                              language="python"
                              dark
                              code={`def score_accessibility(params, overrides=None):
    text = params.text.strip()
    # 1) pattern hits
    idioms = find_terms(text, RULES["idioms"])
    jargon = find_terms(text, RULES["jargon"])
    ambig  = find_terms(text, RULES["ambiguous_time"])
    poly   = find_terms(text, RULES["polysemy"])
    nums   = find_regex(text, RULES["numeracy_date"])

    # 2) density → sigmoid
    n_tokens = max(1, len(tokenize(text)))
    cw = {"idioms":0.35,"jargon":0.35,"ambiguous_time":0.30,"polysemy":0.10,"numeracy_date":0.10}
    density = (cw["idioms"]*len(idioms)+cw["jargon"]*len(jargon)+cw["ambiguous_time"]*len(ambig)+
               cw["polysemy"]*len(poly)+cw["numeracy_date"]*len(nums)) / n_tokens
    center, k = 0.35, 6.0
    base = 1/(1+math.exp(-k*(density-center)))
    prob = min(0.9, base)  # + audience soften

    # 3) explanations + local rewrite
    confusion = build_confusion_sources(idioms, jargon, ambig, poly, nums)
    rewrite = rewrite_15(text, RULES.get("idiom_rewrites", {}))
    bucket = bucket_from_prob(prob, RULES["density_thresholds"])
    return PredictResponse(misinterpretation_probability=prob, risk_bucket=bucket,
                           confusion_sources=confusion, rewrite_15_words=rewrite)`}
                            />
                          </Section>

                          <Section title="Example (UI shows LLM rewrite)">
                            <CodeBlock
                              language="json"
                              dark={false}
                              code={`{
  "model": "accessibility_risk",
  "version": "v1",
  "misinterpretation_probability": 0.18,
  "risk_bucket": "Low",
  "confusion_sources": [{"type":"Idioms/Colloquialisms","evidence":["heads up"]}],
  "rewrite_15_words": "Note: payment is due by September 15." // LLM-polished
}`}
                            />
                          </Section>

                          <Section title="Why leave the probability in (for now)">
                            Even if it’s conservative, the score provides
                            continuity and a place to plug in a trained model
                            later. I want to calibrate it (or replace it) once I
                            have labeled data. Until then, it’s useful for
                            bucketing and for Databot context.
                          </Section>
                        </>
                      )}

                      {tab === "earnings" && (
                        <>
                          <h3 className="text-base font-semibold text-gray-900 mb-2">
                            College Earnings Predictor — Development Notes
                          </h3>

                          <Section title="Goal">
                            Predict median earnings of graduates using
                            institution, major (CIP4), degree level, state, and
                            sector. Specifically, predict the percent chance the
                            graduate will make $75K annually 5 years after
                            graduation.
                          </Section>

                          <Section title="Challenge" as="ul">
                            <li>
                              <strong>Large dataset</strong> (College Scorecard)
                              with millions of rows.
                            </li>
                            <li>
                              <strong>Heavy preprocessing/training</strong> not
                              feasible on Heroku CPU-only dynos.
                            </li>
                            <li>
                              Avoided extra cost/complexity of GPU or hosted
                              training (e.g., Northflank).
                            </li>
                          </Section>

                          <Section title="Solution" as="ol">
                            <li>
                              <strong>Offline training (Jupyter)</strong> —
                              cleaned, encoded, engineered features, selected
                              model (scikit-learn).
                            </li>
                            <li>
                              <strong>Exported artifact</strong> via{" "}
                              <code>joblib</code> to <strong>AWS S3</strong>.
                            </li>
                            <li>
                              <strong>Lightweight inference</strong> — FastAPI
                              loads artifact → instant predictions → React
                              displays results.
                            </li>
                          </Section>

                          <Section title="Why this works" as="ul">
                            <li>
                              <strong>Separation of concerns</strong>: train
                              offline, infer online.
                            </li>
                            <li>
                              <strong>Performance</strong>: no retraining per
                              request.
                            </li>
                            <li>
                              <strong>Cost-efficient</strong>: no GPU/large
                              dynos required.
                            </li>
                            <li>
                              <strong>Maintainable</strong>: update artifact
                              without redeploying the app.
                            </li>
                          </Section>

                          <Section title="Tech stack">
                            Pandas, scikit-learn, Jupyter Notebook, joblib, AWS
                            S3, FastAPI, React.
                          </Section>

                          <Section title="Architecture flow">
                            <CodeBlock
                              language="text"
                              dark={false}
                              code={`[Jupyter Notebook]
    |  (train, evaluate, export .joblib)
    v
   [AWS S3]  <-- artifact storage
    |
    v
[FastAPI Backend] -- loads model --> predict_proba(...)
    |
    v
[React Frontend] -- shows prob, bucket, drivers`}
                            />
                          </Section>

                          <Section title="Snippet — Training (Notebook → artifact)">
                            <CodeBlock
                              code={earnTrain}
                              language="python"
                              dark
                            />
                          </Section>

                          <Section title="Snippet — Inference (Service)">
                            <CodeBlock
                              code={earnInfer}
                              language="python"
                              dark
                            />
                          </Section>

                          <Section title="Example output">
                            <CodeBlock
                              code={earnSample}
                              language="json"
                              dark={false}
                            />
                          </Section>
                        </>
                      )}
                    </div>
                    {/* /Body */}
                  </Dialog.Panel>
                </Transition.Child>
              </div>
              {/* /right-side sheet container */}
            </div>
            {/* /absolute inset-0 overflow-hidden */}
          </div>
          {/* /fixed inset-0 */}
        </Dialog>
      </Transition.Root>
    </>
  );
}
