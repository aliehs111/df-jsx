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

  /* =========================
   * SNIPPETS (what I actually changed)
   * ========================= */

  // 1) FastAPI: cleaned-datasets advisor (summarizes per-dataset metadata and proposes targets)
  const beAdvisor = `# server/routers/databot.py (excerpt)
@router.get("/cleaned_datasets/recommendations")
async def cleaned_datasets_recommendations(task: str = Query(...), db: AsyncSession = Depends(get_async_db)):
    """
    task ∈ {"logistic","multiclass","cluster"}.
    Returns a shortlist of cleaned datasets with badges (mostly_numeric, categorical_features, text_features),
    signals (counts), and candidate target columns by class cardinality.
    """
    # 1) fetch cleaned datasets
    # 2) load per-dataset column stats you computed during cleaning (dtype, n_unique, nulls)
    # 3) derive badges/signals and candidates
    # 4) return items sorted by a simple heuristic (e.g., smaller, cleaner first)
    return {"task": task, "items": items}`;

  // 2) Models.jsx: detect user intent (logistic vs multiclass vs cluster), with simple disambiguation
  const feDetectTask = `// Models.jsx (helpers)
function detectTaskFromQuestion(q) {
  const s = (q || "").toLowerCase();
  const mentionsLogistic = /(logistic|sigmoid)\\b/.test(s);
  const mentionsRF = /(random forest|\\brf\\b|multiclass|multi-class)/.test(s);
  const mentionsCluster = /(cluster|clustering|kmeans|k-means|pca)/.test(s);
  const mentionsBinary = /\\bbinary\\b|\\byes\\b|\\bno\\b|\\btrue\\b|\\bfalse\\b/.test(s);
  if (mentionsLogistic && mentionsRF) return mentionsBinary ? "logistic" : "multiclass";
  if (mentionsLogistic) return "logistic";
  if (mentionsRF) return "multiclass";
  if (mentionsCluster) return "cluster";
  return null;
}`;

  // 3) Models.jsx: call the advisor with a timeout (so chat never hangs)
  const feFetchAdvisor = `async function fetchAdvisor(API_BASE, task) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 4000); // timeout fallback
  try {
    const url = \`\${API_BASE}/api/databot/cleaned_datasets/recommendations?task=\${encodeURIComponent(task)}\`;
    const res = await fetch(url, { method: "GET", credentials: "include", signal: ctrl.signal });
    if (!res.ok) throw new Error(\`advisor \${res.status}\`);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}`;

  // 4) Models.jsx: compact summary to show in chat + prepend into Databot prompt
  const feSummarize = `function summarizeAdvisorResult(advisorJson) {
  const items = advisorJson?.items || [];
  if (!items.length) return "Advisor: No matching cleaned datasets found.";
  const lines = items.slice(0, 8).map((it) => {
    const badges = (it.badges || []).slice(0, 3).join(", ");
    const why = (it.why || [])[0] || "";
    const b = it.candidates?.binary?.slice?.(0, 2) || [];
    const m = it.candidates?.multiclass?.slice?.(0, 2) || [];
    const cand = b.length ? \` (binary: \${b.join(", ")})\` : (m.length ? \` (multiclass: \${m.join(", ")})\` : "");
    return \`• \${it.title}\${badges ? \` — [\${badges}]\` : ""}\${why ? \` — \${why}\` : ""}\${cand}\`;
  });
  return [\`Advisor results for task "\${advisorJson.task}":\`, ...lines].join("\\n");
}`;

  // 5) Databot glue (Databot.jsx): on /models, prepend advisor summary + nudge the LLM to pick one dataset
  const feAskDatabotBranch = `// Databot.jsx (inside askDatabot)
if (location.pathname === "/models") {
  const task = detectTaskFromQuestion(question);
  if (task) {
    try {
      const advisor = await fetchAdvisor(API_BASE, task);
      const summary = summarizeAdvisorResult(advisor);

      // Show shortlist immediately
      setMessages(prev => [...prev, { role: "assistant", content: summary }]);

      // Send to welcome endpoint with deterministic tie-break + safety nudges
      url = \`\${API_BASE}/api/databot/query_welcome\`;
      question = \`\${summary}

System: From these advisor results, recommend the single best dataset and, if supervised, the exact target column to use.
If multiple candidates tie, prefer the smaller dataset (faster iteration).
If no binary candidate is found, suggest a likely target by name and why.
Be concise (<120 words).

User question: \${userMessage.content}\`;
      payload = { question, app_info: appInfo };
    } catch (e) {
      // Timeout or advisor error → graceful fallback
      url = \`\${API_BASE}/api/databot/query_welcome\`;
      question = \`User asked about dataset-model fit on the Models page. (Advisor failed: \${e?.message || e})
Question: \${userMessage.content}\`;
      payload = { question, app_info: appInfo };
    }
  }
}`;

  // 6) Models.jsx: lightweight badges under each dataset title (fed by advisor)
  const feBadges = `// Models.jsx (rendering inside the dataset list item)
{datasetBadges[ds.id]?.badges?.length > 0 && (
  <div className="mt-2 flex flex-wrap gap-1" title={datasetBadges[ds.id]?.hint || ""}>
    {datasetBadges[ds.id].badges.map((b, i) => (
      <span
        key={i}
        className="inline-flex items-center rounded-full bg-gray-100 text-gray-700 px-2 py-0.5 text-[10px] font-semibold ring-1 ring-gray-300"
      >
        {b}
      </span>
    ))}
  </div>
)}`;

  return (
    <>
      {/* Floating button */}
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
                        Models — Dev Notes (for Instructor)
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
                      <Section title="What this page does">
                        The <strong>Models</strong> page lets me choose a
                        cleaned dataset, pick a model (Random Forest, Logistic
                        Regression, PCA+KMeans, Sentiment, Anomaly Detection,
                        Time Series), and run it. The hard part here was{" "}
                        <strong>advising which dataset fits which model</strong>
                        using <em>multiple datasets’ metadata at once</em>.
                      </Section>

                      <Section title="Challenges">
                        Initially, Databot only saw the selected dataset. To
                        give useful advice (e.g., “Use dataset A with target X
                        for logistic regression”), it needed a
                        <strong> cross-dataset view</strong> of metadata
                        (dtypes, unique counts, text/categorical mix). I added a
                        small “advisor” API that aggregates cleaned dataset
                        metadata and returns a shortlist with badges and target
                        candidates. Then I wired Databot to consult that
                        shortlist when the user asks model-selection questions
                        on this page.
                      </Section>

                      <Section title="What I built (high level)" as="ol">
                        <li>
                          <strong>Advisor API (FastAPI):</strong> returns a
                          ranked list of cleaned datasets with badges (
                          <code>mostly_numeric</code>,{" "}
                          <code>categorical_features</code>,{" "}
                          <code>text_features</code>), signals (counts), and
                          likely target columns (binary/multiclass).
                        </li>
                        <li>
                          <strong>Databot glue on /models:</strong> detects user
                          intent (logistic vs multiclass vs cluster), calls the
                          advisor with a timeout, shows a compact shortlist,
                          then prepends it to the LLM prompt with instructions
                          to pick exactly one dataset + target and give 3 setup
                          steps.
                        </li>
                        <li>
                          <strong>Badges in UI:</strong> small chips under each
                          dataset name so users see the metadata signals at a
                          glance.
                        </li>
                      </Section>

                      <Section title="Backend: advisor endpoint (FastAPI)">
                        <CodeBlock language="python" dark code={beAdvisor} />
                      </Section>

                      <Section title="Frontend: detect task intent">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feDetectTask}
                        />
                      </Section>

                      <Section title="Frontend: advisor fetch with timeout">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feFetchAdvisor}
                        />
                      </Section>

                      <Section title="Frontend: shortlist summary + Databot prompt nudge">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feSummarize}
                        />
                        <div className="h-2" />
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feAskDatabotBranch}
                        />
                      </Section>

                      <Section title="Frontend: badges on dataset list">
                        <CodeBlock language="javascript" dark code={feBadges} />
                      </Section>

                      <Section title="How Databot uses this">
                        On the <code>/models</code> route, Databot first shows
                        the advisor’s shortlist (so the user sees the options),
                        then the LLM answers with a{" "}
                        <strong>single recommendation</strong> and{" "}
                        <strong>3 setup steps</strong>
                        (e.g., imputation/encoding/validation). This gives
                        actionable guidance without the user needing to inspect
                        every dataset manually.
                      </Section>

                      <Section title="Limitations / next step">
                        <ul className="list-disc list-inside space-y-1">
                          <li>
                            For time constraints, I didn’t finish an
                            auto-explain step where Databot reads the returned
                            model metrics (e.g., confusion matrix, feature
                            importances) and{" "}
                            <strong>summarizes results in plain English</strong>
                            . If I had more time, I’d add that as a follow-up
                            prompt using the model run JSON.
                          </li>
                          <li>
                            Advisor is heuristic (simple signals from metadata).
                            It works well for this class project size (≤20
                            datasets).
                          </li>
                        </ul>
                      </Section>

                      <Section title="Operational note (GPU)">
                        ⚡ GPU inference is{" "}
                        <strong>available by request</strong> (I manually enable
                        it for demos). The page includes a “Request GPU Access”
                        button that opens an email; responses may be delayed.
                      </Section>

                      <Section title="Takeaway">
                        Getting Databot to advise usefully required{" "}
                        <strong>multi-dataset context</strong>. Once I built a
                        small advisor and passed its summary into the chat
                        prompt on this page, the recommendations became concrete
                        (dataset + target + steps). That was the main unlock.
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
