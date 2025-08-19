// client/src/components/DevNotesDatabot.jsx
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

export default function DevNotesDatabot() {
  const [open, setOpen] = useState(false);

  /* =========================
   * SNIPPETS (copyable)
   * ========================= */

  // A) FE events from DataCleaning → Databot (what you emit)
  const feEvents = `// DataCleaning.jsx (already wired)
// 1) Rich context for Databot (plan + deltas)
window.dispatchEvent(new CustomEvent("databot:context", { detail: {
  datasetId: Number(id),
  page: "data-cleaning",
  intent,                 // "options_updated" | "preview" | "preview_result" | "save_request" | "save_success" | "save_error"
  summary,                // summarizePipeline(pipeline)
  result_summary,         // computeStatsDelta(...).summary
  before_stats, after_stats,
  alerts,
  selectedOpsSummary: summary
}}));

// 2) One-line breadcrumb for "what just happened"
window.dispatchEvent(new CustomEvent("dfjsx-cleaning-action", {
  detail: { action: "Filled missing values in 'age' with median" }
}));`;

  // B) Databot.jsx: route key helper used in prompts/telemetry
  const feRouteKey = `// Databot.jsx
function routeToPageKey(pathname) {
  if (!pathname) return "unknown";
  if (pathname === "/dashboard") return "dashboard";
  if (pathname === "/models") return "models/index";
  if (pathname === "/predictors") return "predictors/index";
  if (/^\\/datasets\\/\\d+$/.test(pathname)) return "datasets/detail";
  if (/^\\/datasets\\/\\d+\\/clean/.test(pathname)) return "data-cleaning";
  return "other";
}`;

  // C) Databot.jsx: advisor fetch with timeout (used on /models)
  const feAdvisor = `async function fetchAdvisor(base, task) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 4000);
  try {
    const u = \`\${base}/api/databot/cleaned_datasets/recommendations?task=\${encodeURIComponent(task)}\`;
    const res = await fetch(u, { method: "GET", credentials: "include", signal: ctrl.signal });
    if (!res.ok) throw new Error("advisor " + res.status);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}`;

  // D) Databot.jsx: branch map (what context is used where)
  const feBranches = `// Databot.jsx (inside askDatabot)
// 1) /dashboard → static appInfo via query_welcome
// 2) /models → detectTaskFromQuestion → advisor → prepend shortlist → query_welcome
// 3) /predictors → modelbot header built from forcedContext (feature, inputs, result)
// 4) /datasets/:id/clean → prepend cleaningContext (plan, lastAction, resultSummary) and dataset_id
// 5) default → send dataset_id and app_info to /api/databot/query
// All paths include credentials: "include" so server-side session can read DB context (columns, targets, etc).`;

  // E) Backend: ask endpoint (schema-first)
  const beAsk = `# server/routers/databot.py (essentials)
@router.post("/ask")
async def ask(req: AskRequest, db: AsyncSession = Depends(get_async_db)):
    ctx = await build_context(db=db, page=req.page, dataset_id=req.dataset_id, hints=req.context_hints or [])
    prompt, rev = render_prompt(req.page, req.user_message, ctx)
    out = await call_llm(prompt=prompt, temperature=0.2, max_tokens=600, timeout_s=20)
    if not out.ok:
        raise HTTPException(status_code=502, detail=f"LLM error: {out.reason}")
    return {
        "answer": out.text,
        "used_context": ctx.public_dict(),
        "model": out.model,
        "prompt_revision": rev,
        "latency_ms": out.latency_ms,
        "input_tokens": out.usage.prompt_tokens,
        "output_tokens": out.usage.completion_tokens,
        "request_id": out.request_id,
    }`;

  // F) Prompt: dataset/detail and data-cleaning variants
  const bePrompts = `# server/routers/prompts.py (sketch)
def render_prompt(page: str, user: str, ctx):
    if page == "datasets/detail":
        return (
            "\\n".join([
                "You are a data science tutor. Be concrete and concise.",
                f"dataset_id: {getattr(ctx, 'dataset_id', None)}",
                f"columns: {getattr(ctx, 'columns', [])}",
                f"target: {getattr(ctx, 'target', None)}",
                f"n_rows: {getattr(ctx, 'n_rows', 'unknown')}",
                f"missing_values: {getattr(ctx, 'has_missing_values', False)}",
                f"User: {user}",
            ]),
            "v3"
        )
    if page == "data-cleaning":
        return (
            "\\n".join([
                "You are a data cleaning tutor. Explain what just changed and why it matters.",
                f"dataset_id: {getattr(ctx, 'dataset_id', None)}",
                f"plan: {getattr(ctx, 'plan', '-')}",
                f"delta: {getattr(ctx, 'result_summary', '-')}",
                "Rules: keep under 100 words.",
                f"User: {user}",
            ]),
            "v2"
        )
    # default
    return (f"Page: {page}\\nHints: {getattr(ctx, 'hints', [])}\\nUser: {user}", "v1")`;

  // G) Context builder adds cleaning hints when page == data-cleaning
  const beContext = `# server/routers/context.py (sketch)
async def build_context(db, page, dataset_id, hints):
    ctx = {"page": page, "hints": hints}
    if dataset_id:
        ds = await load_dataset_row(db, dataset_id)
        if ds:
            ctx |= {
                "dataset_id": ds.id,
                "columns": list((ds.column_metadata or {}).keys()),
                "target": ds.target_column,
                "n_rows": ds.n_rows,
                "n_columns": ds.n_columns,
                "has_missing_values": ds.has_missing_values,
            }
    if page == "data-cleaning":
        # these are posted from the FE as part of state sync
        last = await read_last_cleaning_state(db, dataset_id)
        if last:
            ctx |= {
                "plan": last.last_summary,
                "result_summary": last.last_result,
            }
    return SimpleNamespace(**ctx, public_dict=lambda: ctx)`;

  // H) Troubleshooting checklist
  const troubleshooting = `• If Databot answers feel generic on /models, verify advisor endpoint returns items and that Databot.jsx is on the query_welcome path for that route.
• If cleaning context is ignored, confirm the FE emits "databot:context" and "dfjsx-cleaning-action", and that Databot.jsx listens on those events (setForcedContext + cleaningContext).
• If dataset-specific facts are missing, check credentials: "include" on fetch and your server session/DB access.
• If chat freezes on advisor calls, confirm fetchWithTimeout and the 4s abort are in place (fallback to normal welcome path).
• For old datasets uploaded pre-change, refresh metadata (or re-open the page) so the backend context aligns with the new schema.`;

  return (
    <>
      {/* Floating button (bottom-left) */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-[60] rounded-full bg-accent p-3 shadow-lg hover:bg-accent/90 text-white"
        title="Databot — Dev Notes"
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
                        Dev Notes — Databot
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
                      <Section title="Summary">
                        Each chat request builds page-aware context, renders a
                        versioned prompt, calls the LLM with guardrails, and
                        returns a typed payload with telemetry. Context sources
                        vary by route: app info (dashboard), dataset metadata
                        (detail), in-flight cleaning state (data-cleaning),
                        cross-dataset advisor (models), or result bundles
                        (predictors).
                      </Section>

                      <Section title="Frontend events from DataCleaning">
                        <CodeBlock language="javascript" dark code={feEvents} />
                      </Section>

                      <Section title="Databot routing overview (frontend)">
                        <CodeBlock language="text" dark code={feBranches} />
                        <div className="h-2" />
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feRouteKey}
                        />
                      </Section>

                      <Section title="Models advisor: timeout + fetch">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feAdvisor}
                        />
                      </Section>

                      <Section title="Backend: ask endpoint (schema-first)">
                        <CodeBlock language="python" dark code={beAsk} />
                      </Section>

                      <Section title="Prompts (dataset detail & cleaning)">
                        <CodeBlock language="python" dark code={bePrompts} />
                        <div className="h-2" />
                        <CodeBlock language="python" dark code={beContext} />
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
