// client/src/components/DevNotesModels.jsx
import { useState, Fragment } from "react";
import { Dialog, Transition } from "@headlessui/react";
import {
  XMarkIcon,
  InformationCircleIcon,
  ClipboardIcon,
  CheckIcon,
} from "@heroicons/react/24/outline";

/* ---------- Helper: uniform section ---------- */
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

/* ---------- Helper: code block with copy ---------- */
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

export default function DevNotesModels() {
  const [open, setOpen] = useState(false);

  // Frontend payload example (generic runner)
  const fePayload = `// Frontend -> POST /api/models/run
const payload = {
  dataset_id,
  model: selectedModel, // "RandomForest" | "LogisticRegression" | "PCA_KMeans" | "AnomalyDetection" | "TimeSeriesForecasting" | "Sentiment"
  params: {
    target: selectedTarget || null,
    n_estimators,
    max_depth,
    C,
    n_clusters,
    date_column,
    value_column
  }
};

const res = await fetch("/api/models/run", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  credentials: "include",
  body: JSON.stringify(payload),
});
const result = await res.json();
setResult(result);`;

  // Backend model switcher (FastAPI service layer)
  const beSwitch = `# server/models/runner.py
from .rf import run_random_forest
from .logreg import run_logistic_regression
from .pca_kmeans import run_pca_kmeans
from .anomaly import run_anomaly_detection
from .ts_forecast import run_time_series
from .sentiment import run_sentiment

def run_model(dataset, model_name, params):
    if model_name == "RandomForest":
        return run_random_forest(dataset, params)
    if model_name == "LogisticRegression":
        return run_logistic_regression(dataset, params)
    if model_name == "PCA_KMeans":
        return run_pca_kmeans(dataset, params)
    if model_name == "AnomalyDetection":
        return run_anomaly_detection(dataset, params)
    if model_name == "TimeSeriesForecasting":
        return run_time_series(dataset, params)
    if model_name == "Sentiment":
        return run_sentiment(dataset, params)
    raise ValueError(f"Unknown model: {model_name}")`;

  // Example packaging for classification metrics
  const beMetrics = `# server/models/logreg.py (excerpt)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, pandas as pd

def run_logistic_regression(dataset, params):
    X, y = prepare_features_and_target(dataset, params["target"])
    clf = build_logreg(params)  # hyperparams -> estimator
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "model": "LogisticRegression",
        "classification_report": report,
        "confusion_matrix": cm,
        "class_counts": pd.Series(y).value_counts().to_dict()
    }`;

  // Example result JSON
  const exampleJson = `{
  "model": "RandomForest",
  "message": "fit complete",
  "class_counts": {"0": 124, "1": 76},
  "classification_report": {
    "0": {"precision": 0.86, "recall": 0.83, "f1-score": 0.84, "support": 124},
    "1": {"precision": 0.75, "recall": 0.79, "f1-score": 0.77, "support": 76},
    "accuracy": 0.82,
    "macro avg": {"precision": 0.80, "recall": 0.81, "f1-score": 0.81, "support": 200},
    "weighted avg": {"precision": 0.82, "recall": 0.82, "f1-score": 0.82, "support": 200}
  }
}`;

  return (
    <>
      {/* Floating button — bottom-left to match Predictors */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-[60] rounded-full bg-accent p-3 shadow-lg hover:bg-accent/90 text-white"
        title="Models — Dev Notes"
        aria-label="Open Dev Notes"
      >
        <InformationCircleIcon className="h-6 w-6" />
      </button>

      {/* Right slide-over */}
      <Transition.Root show={open} as={Fragment}>
        <Dialog as="div" className="relative z-[999]" onClose={setOpen}>
          {/* Overlay */}
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

          {/* Sheet */}
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
                      <Dialog.Title className="text-lg font-semibold text-gray-900">
                        Models — Dev Notes
                      </Dialog.Title>
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
                      <h3 className="text-base font-semibold text-gray-900">
                        Page-Level Model Runner — Notes
                      </h3>

                      <Section title="Goal">
                        Provide a unified UI to configure and execute multiple
                        ML models on a cleaned dataset, and return typed results
                        (metrics, plots, summaries) suitable for rendering in
                        React.
                      </Section>

                      <Section title="Contract (frontend → backend)" as="ul">
                        <li>
                          <strong>Endpoint:</strong>{" "}
                          <code className="font-mono">
                            POST /api/models/run
                          </code>
                        </li>
                        <li>
                          <strong>Payload:</strong>{" "}
                          <code className="font-mono">{`{ dataset_id, model, params }`}</code>
                        </li>
                        <li>
                          <strong>Response:</strong> JSON typed by model (e.g.,
                          classification report, confusion matrix, PCA variance,
                          anomalies, image_base64)
                        </li>
                      </Section>

                      <Section title="Frontend payload example">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={fePayload}
                        />
                      </Section>

                      <Section title="Backend model dispatcher">
                        <CodeBlock language="python" dark code={beSwitch} />
                      </Section>

                      <Section title="Packaging metrics (example: Logistic Regression)">
                        <CodeBlock language="python" dark code={beMetrics} />
                      </Section>

                      <Section title="Example result JSON">
                        <CodeBlock
                          language="json"
                          dark={false}
                          code={exampleJson}
                        />
                      </Section>

                      <Section title="Notes">
                        <ul className="list-disc list-inside space-y-1">
                          <li>
                            <strong>PCA+KMeans:</strong> returns{" "}
                            <code>pca_variance_ratio</code> and{" "}
                            <code>cluster_counts</code>.
                          </li>
                          <li>
                            <strong>RandomForest:</strong> returns{" "}
                            <code>feature_importances</code>; consider limiting
                            top-k for UI.
                          </li>
                          <li>
                            <strong>TimeSeries:</strong> validates{" "}
                            <code>date_column|value_column</code>; errors if not
                            provided.
                          </li>
                          <li>
                            <strong>Sentiment:</strong> expects a text column;
                            returns <code>sentiment_counts</code> + sample
                            results.
                          </li>
                          <li>
                            <strong>Error paths</strong> are normalized so the
                            UI can map common issues (e.g., missing target,
                            empty dataset).
                          </li>
                        </ul>
                      </Section>
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
