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
            <span className="font-medium">Heuristic Scorer</span>: Message
            Clarity & Misinterpretation Risk
            <span className="text-gray-500">
              (rule-based + OpenAI API responses, so to run on CPU in deployed
              environment without GPU)
            </span>
          </li>
          <li>
            <span className="font-medium">ML Model</span>: College Earnings
            Potential
            <span className="text-gray-500">
              (logistic regression trained offline in Jupyter; artifacts bundled
              with the repo)
            </span>
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
            <span className="font-medium">Deployments</span>: Heroku web +
            worker dynos for APIs and tasks; optional Northflank service for
            heavier or GPU-based inference.
          </li>
          <li>
            <span className="font-medium">Environments & Secrets</span>:
            <code>.env</code> for local; Heroku config vars for production (DB
            URL, S3 creds, JWT, CORS).
          </li>
          <li>
            <span className="font-medium">Model Registry</span>: Versioned
            models and plots stored on AWS S3 with predictable keys (e.g.{" "}
            <code>
              models/{"{"}model{"}"}/{"{"}version{"}"}/
            </code>
            ).
          </li>
          <li>
            <span className="font-medium">CI/CD</span>: Integrated with Heroku
            pipelines. Post-deploy health checks verify endpoints and confirm
            models load correctly before going live.
          </li>
          <li>
            <span className="font-medium">Version Tracking</span>: Training
            reports and model artifacts link back to git commit hashes, making
            builds easy to trace and reproduce.
          </li>
          <li>
            <span className="font-medium">Observability</span>: Structured JSON
            logging from FastAPI with request IDs, timing, and inference latency
            metrics.
          </li>
          <li>
            <span className="font-medium">Error Handling & Retries</span>:
            Graceful API fallbacks for 4xx/5xx responses; client retry/backoff
            for transient errors.
          </li>
          <li>
            <span className="font-medium">Data Lifecycle</span>: In-memory temp
            processing; finalized datasets stored on S3; optional lifecycle
            policies for archiving.
          </li>
          <li>
            <span className="font-medium">Reproducibility</span>: Pinned
            dependencies, environment snapshots, and model version tags surfaced
            in the UI.
          </li>
          <li>
            <span className="font-medium">Database</span>: MySQL (JawsDB) for
            metadata, parameters, and run history; async SQLAlchemy; protected
            routes.
          </li>
          <li>
            <span className="font-medium">Security</span>: CORS allowlist, JWT
            authentication, and presigned S3 URLs for controlled downloads.
          </li>
          <li>
            <span className="font-medium">Performance</span>: Batch endpoints,
            streaming for long-running tasks, and queue-ready worker patterns.
          </li>
          <li>
            <span className="font-medium">Backups & Recovery</span>: DB
            snapshots plus S3 object versioning for model artifacts.
          </li>
        </ul>
      </Panel>

      {/* Chatbot */}
      <Panel title="Chatbot">
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-2">
          <li>
            <span className="font-medium">Context-Aware Assistant</span>:
            Explains data cleaning steps and model prediction output in plain
            language, translating technical aspects into user-friendly insights.
          </li>
          <li>
            <span className="font-medium">Page-Specific Guidance</span>:
            Provides tailored prompts depending on where the user is in the app.
            The context is conditional based on component, active files and user
            actions where applicable.
          </li>
          <li>
            <span className="font-medium">Actionable Hints</span>: Suggests next
            steps such as <em>impute vs. drop</em> or
            <em>scale vs. standardize</em>, reducing trial-and-error.
          </li>
          <li>
            <span className="font-medium">API-First Design</span>: Chatbot runs
            as a lightweight API service so it can evolve independently of the
            UI, making it easy to update or swap models without rewriting
            components.
          </li>
          <li>
            <span className="font-medium">Explainability Layer</span>: Suggests
            Model/Dataset pairing for user based on metadata from datasets and
            models. Answers user's questions as it applies to current dataset.
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
            Understand that df.jsx is learning project and not ready for prime
            time.
          </li>
        </ul>
      </Panel>
    </div>
  );
}
