import { Disclosure } from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/24/outline";

function Panel({ title, children }) {
  return (
    <Disclosure as="div" className="rounded-lg border border-gray-200 bg-white">
      {({ open }) => (
        <>
          <Disclosure.Button className="flex w-full items-center justify-between px-4 py-3 text-left">
            <span className="text-sm font-semibold text-gray-900">{title}</span>
            <ChevronDownIcon
              className={`h-5 w-5 transition-transform ${
                open ? "rotate-180" : ""
              }`}
            />
          </Disclosure.Button>
          <Disclosure.Panel className="px-5 pb-5">{children}</Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}

export default function FeatureDropdowns() {
  return (
    <div className="space-y-4">
      {/* Data Analysis */}
      <Panel title="Data Analysis">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            CSV upload → instant preview (<code>head()</code>),{" "}
            <code>shape</code>, and <code>info()</code>; nothing is persisted
            until you choose to save.
          </li>
          <li>
            EDA tools: summary stats, missing-value report, correlation heatmap,
            and automatic column type detection (numeric / categorical /
            datetime).
          </li>
          <li>
            Cleaning pipeline with before/after metrics: missing-value handling
            (drop/fill mean/median/mode/custom), scaling
            (normalize/standardize), and categorical encoding (one-hot/label).
          </li>
          <li>
            Dataset metadata tracking: row/column counts, null indicators,
            inferred schema, target column & selected features, plus a stepwise
            processing log.
          </li>
          <li>
            Exports: cleaned dataset and generated visuals via presigned S3
            downloads.
          </li>
        </ul>
      </Panel>

      {/* Modeling */}
      <Panel title="Modeling">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            <span className="font-medium">Predictors</span>: Accessibility
            Misinterpretation Risk (OpenAI-backed responses to avoid GPU) and
            College Earnings Potential (offline-trained in a notebook; artifacts
            bundled with the repo).
          </li>
          <li>
            <span className="font-medium">Models (fit on your data)</span>:
            Random Forest, Logistic Regression, PCA&nbsp;+&nbsp;KMeans, Anomaly
            Detection, and Feature Importance.
          </li>
          <li>
            Configurable parameters where appropriate (e.g.,{" "}
            <code>n_estimators</code>, <code>max_depth</code>, <code>C</code>,{" "}
            <code>n_components</code>, <code>k</code>) with immediate results
            after fit.
          </li>
          <li>
            Compute paths: CPU by default; GPU-backed workloads are
            transparently offloaded to Northflank.
          </li>
          <li>
            Results in clean, card-based layouts—metrics, sample predictions,
            cluster counts, variance explained—with plots rendered via
            Matplotlib and served as static images.
          </li>
        </ul>
      </Panel>

      {/* DevOps / MLOps */}
      <Panel title="DevOps / MLOps">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            Deployments: Heroku web + worker dynos for API/tasks; optional
            Northflank service for heavier or GPU-friendly inference endpoints
          </li>
          <li>
            Environments & secrets: .env for local, Heroku config vars for prod
            (DB URL, S3 creds, JWT, CORS)
          </li>
          <li>
            Build artifacts & model registry: versioned models and plots stored
            on AWS S3 with predictable keys (e.g.,{" "}
            <code>
              models/{"{"}model{"}"}/{"{"}version{"}"}/
            </code>
            )
          </li>
          <li>
            CI/CD hooks (optional): push-to-deploy & health check route for
            smoke tests
          </li>
          <li>
            Observability: structured JSON logging from FastAPI; request IDs,
            timing, and inference latency metrics
          </li>
          <li>
            Error handling & retries: graceful API fallbacks for 4xx/5xx; client
            retry/backoff for transient errors
          </li>
          <li>
            Data lifecycle: temp in-memory processing; finalized datasets to S3;
            optional S3 lifecycle policies
          </li>
          <li>
            Reproducibility: pinned deps (requirements/conda), environment
            snapshots, and model version tags shown in UI
          </li>
          <li>
            Database: MySQL (JawsDB) for metadata/params/run history; async
            SQLAlchemy; protected routes
          </li>
          <li>
            Security: CORS allowlist, JWT, and presigned S3 URLs for controlled
            downloads
          </li>
          <li>
            Performance: batch endpoints, streaming for long tasks, queue-ready
            worker patterns
          </li>
          <li>
            Backups & recovery: DB snapshots + S3 object versioning for
            artifacts
          </li>
        </ul>
      </Panel>

      {/* Chatbot */}
      <Panel title="Chatbot">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            Context-aware Databot explains cleaning steps and model outputs in
            plain language
          </li>
          <li>
            Page-aware guidance (e.g., different prompts on Predictors vs. Data
            Cleaning)
          </li>
          <li>
            Action hints: suggests next steps (impute vs. drop, scale vs.
            standardize, etc.)
          </li>
          <li>
            Lightweight, API-first integration so chatbot can evolve without UI
            rewrites
          </li>
        </ul>
      </Panel>
      {/* DevNotes */}
      <Panel title="Development Notes">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            The green buttons on the bottom left of the pages open panels that
            briefly describe some development notes and relevant code snippets.
          </li>
          <li>
            To see the whole code base, please visit my repo at{" "}
            <a
              href="https://github.com/aliehs111/df-jsx"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline hover:text-blue-700"
            >
              github.com/aliehs111/df-jsx
            </a>
            .
          </li>
          <li>
            If you have any questions or comments, please see the feedback
            button on the bottom of this page to email me.
          </li>
          <li>
            Understand that df.jsx is strictly an academic experiment and not
            ready for prime time.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
