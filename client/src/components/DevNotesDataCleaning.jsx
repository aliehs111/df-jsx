// client/src/components/DevNotesDataCleaning.jsx
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

export default function DevNotesDataCleaning() {
  const [open, setOpen] = useState(false);

  /* =========================
   * SNIPPETS (focused extracts)
   * ========================= */

  // A) Push state to backend + broadcast context to Databot (frontend)
  const fePushState = [
    "// DataCleaning.jsx (inside pushDatabotState)",
    "const plan = summary || (pipeline ? summarizePipeline(pipeline) : '');",
    "const assistant_hints = [",
    "  intent ? 'Intent: ' + intent : null,",
    "  plan ? 'Plan:\\n' + plan : null,",
    "  result_summary ? 'Result:\\n' + result_summary : null,",
    "  options.lowercase_headers ? 'Lowercase headers enabled.' : null,",
    "  options.remove_duplicates ? 'Remove duplicates enabled.' : null,",
    "  options.dropna ? 'Drop NA rows enabled.' : null,",
    "  options.fillna_strategy && options.selected_columns?.fillna?.length",
    "    ? 'FillNA (' + options.fillna_strategy + ') → ' + options.selected_columns.fillna.join(', ')",
    "    : null,",
    "  options.encoding && options.selected_columns?.encoding?.length",
    "    ? 'Encode (' + options.encoding + ') → ' + options.selected_columns.encoding.join(', ')",
    "    : null,",
    "  options.scale && options.selected_columns?.scale?.length",
    "    ? 'Scale (' + options.scale + ') → ' + options.selected_columns.scale.join(', ')",
    "    : null,",
    "].filter(Boolean);",
    "",
    "const body = {",
    "  dataset_id: Number(id),",
    "  options: {",
    "    fillna_strategy: options.fillna_strategy,",
    "    scale: options.scale,",
    "    encoding: options.encoding,",
    "    lowercase_headers: options.lowercase_headers,",
    "    dropna: options.dropna,",
    "    remove_duplicates: options.remove_duplicates,",
    "    outlier_method: options.outlier_method,",
    "    conversions: options.conversions,",
    "    binning: options.binning,",
    "    selected_columns: options.selected_columns,",
    "    assistant_hints,",
    "    intent: intent || undefined,",
    "    last_summary: plan || undefined,",
    "    last_result: result_summary || undefined,",
    "    page: 'data-cleaning',",
    "  },",
    "  before_stats,",
    "  after_stats,",
    "  alerts,",
    "};",
    "",
    "// POST → /api/databot/state/{id} (debounced for UX)",
    "debouncedFetch(backendUrl + '/api/databot/state/' + id, {",
    "  method: 'POST',",
    "  credentials: 'include',",
    "  headers: { 'Content-Type': 'application/json' },",
    "  body: JSON.stringify(body),",
    "}, setAlerts);",
    "",
    "// Also emit a window event Databot.jsx listens to",
    "window.dispatchEvent(new CustomEvent('databot:context', {",
    "  detail: {",
    "    datasetId: Number(id),",
    "    page: 'data-cleaning',",
    "    intent,",
    "    pipeline,",
    "    summary: plan,",
    "    result_summary,",
    "    before_stats,",
    "    after_stats,",
    "    alerts,",
    "    selectedOpsSummary: summarizePipeline(pipeline),",
    '    initial_message: pipeline?.length ? "I see you\'ve made some changes to preview..." : null,',
    "  },",
    "}));",
  ].join("\n");

  // B) Record the last concrete user action (used for Databot 'last action' header)
  const feActionEvent = [
    "// DataCleaning.jsx (after preview/save completes, derive a human label)",
    "const action = op.operation === 'convert' ?",
    '  ("Transformed datatype for \'" + op.column + "\' to " + op.value) :',
    "  op.operation === 'fillna' ?",
    '  ("Filled missing values in \'" + op.column + "\' with " + op.value) :',
    "  op.operation === 'scale' ?",
    '  ("Scaled column \'" + op.column + "\' using " + op.value) :',
    "  op.operation === 'encoding' ?",
    '  ("Encoded column \'" + op.column + "\' using " + op.value) :',
    "  op.operation === 'outliers' ?",
    '  ("Removed outliers in \'" + op.column + "\' using " + op.value) :',
    "  op.operation === 'binning' ?",
    "  (\"Binned column '\" + op.column + \"' into \" + op.value + ' bins') :",
    "  'Applied cleaning action';",
    "",
    "window.dispatchEvent(new CustomEvent('dfjsx-cleaning-action', {",
    "  detail: { action }",
    "}));",
  ].join("\n");

  // C) Databot.jsx: consume events + prepend cleaning context in prompt
  const feDatabotGlue = [
    "// Databot.jsx (listeners)",
    "useEffect(() => {",
    "  const onCtx = (e) => {",
    "    const d = e.detail || {};",
    "    if (d.page !== 'data-cleaning') return;",
    "    setIsOpen(true);",
    "    if (d.intent === 'options_updated' && (d.summary || d.selectedOpsSummary)) {",
    "      const plan = d.summary || d.selectedOpsSummary;",
    "      setCleaningContext((prev) => ({ ...prev, pipelineSummary: plan || '' }));",
    "      setMessages((prev) => [...prev, {",
    "        role: 'assistant',",
    "        content: 'I see you\\'re setting up a cleaning step:\\n\\n' + plan + '\\n\\nWant a quick explanation?',",
    "      }]);",
    "    }",
    "    if (d.intent === 'preview_result' && d.result_summary) {",
    "      setCleaningContext((prev) => ({ ...prev, resultSummary: d.result_summary || '' }));",
    "    }",
    "  };",
    "  window.addEventListener('databot:context', onCtx);",
    "  return () => window.removeEventListener('databot:context', onCtx);",
    "}, []);",
    "",
    "useEffect(() => {",
    "  const onAction = (e) => {",
    "    const a = e.detail?.action || '';",
    "    if (a) { setIsOpen(true); setCleaningContext((p) => ({ ...p, lastAction: a })); }",
    "  };",
    "  window.addEventListener('dfjsx-cleaning-action', onAction);",
    "  return () => window.removeEventListener('dfjsx-cleaning-action', onAction);",
    "}, []);",
    "",
    "// Databot.jsx (askDatabot cleaning branch)",
    "if (!usedAdvisor) {",
    "  if (location.pathname.includes('/cleaning')) {",
    "    const header = [",
    "      cleaningContext?.pipelineSummary ? 'Plan:\\n' + cleaningContext.pipelineSummary : null,",
    "      cleaningContext?.lastAction ? 'Last action:\\n' + cleaningContext.lastAction : null,",
    "      cleaningContext?.resultSummary ? 'Result:\\n' + cleaningContext.resultSummary : null,",
    "    ].filter(Boolean).join('\\n\\n');",
    "    if (header) question = header + '\\n\\n' + question;",
    "    payload = {",
    "      question,",
    "      bot_type: effectiveBotType,",
    "      ...(datasetId != null ? { dataset_id: datasetId } : {}),",
    "    };",
    "  }",
    "}",
  ].join("\n");

  // D) Stats deltas & Coach (front-end reasoning for inline hints)
  const feCoach = [
    "// DataCleaning.jsx (coach + stats delta, excerpts)",
    "function computeStatsDelta(before, after) {",
    "  if (!before || !after) return { summary: 'No stats available.', changedNulls: [] };",
    "  const br = before.shape?.[0], bc = before.shape?.[1];",
    "  const ar = after.shape?.[0], ac = after.shape?.[1];",
    "  const shapeDelta = (br != null && ar != null && bc != null && ac != null)",
    "    ? ('Shape: ' + br + '×' + bc + ' → ' + ar + '×' + ac)",
    "    : 'Shape: unknown';",
    "  // compare null_counts → build human summary",
    "  return { summary: shapeDelta /* + null deltas */ , changedNulls: [] };",
    "}",
    "",
    "function coach(before, after, ops) {",
    "  const msgs = [];",
    "  if (ops.lowercase_headers) msgs.push('Lowercased headers.');",
    "  if (ops.remove_duplicates) msgs.push('Removed duplicate rows.');",
    "  if (ops.dropna) msgs.push('Dropped rows with missing values.');",
    "  // ...plus messages for fillna/encoding/scale/conversions/binning",
    "  return msgs;",
    "}",
  ].join("\n");

  // E) Backend endpoint (shape only; FastAPI)
  const beStateApi = [
    "# server/routers/databot.py (shape of /api/databot/state/{id})",
    "@router.post('/api/databot/state/{dataset_id}')",
    "async def set_state(dataset_id: int, payload: DatabotStateIn):",
    "    # payload.options.assistant_hints (list[str])",
    "    # payload.options.page == 'data-cleaning'",
    "    # payload.before_stats / payload.after_stats, payload.alerts",
    "    # Store transient context so Databot can answer: what changed, why it matters.",
    "    return {'ok': True}",
  ].join("\n");

  // F) Optional: small prime button (forces a gentle “I see your plan” message)
  const fePrimeButton = [
    "// In DataCleaning header actions (example)",
    "<button",
    "  onClick={() => window.dispatchEvent(new CustomEvent('databot:context', {",
    "    detail: {",
    "      page: 'data-cleaning',",
    "      intent: 'options_updated',",
    "      summary: selectedOpsSummary,",
    "      selectedOpsSummary,",
    "      datasetId: Number(id),",
    "    }",
    "  }))}",
    "  className='inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-sm text-white/90 ring-1 ring-white/20 hover:bg-white/15'>",
    "  Prime Databot",
    "</button>",
  ].join("\n");

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-[60] rounded-full bg-emerald-500 p-3 shadow-lg hover:bg-emerald-600 text-white"
        title="Data Cleaning — Dev Notes"
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
                        Data Cleaning — Dev Notes
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
                        The <strong>Data Cleaning</strong> page lets users build
                        a lightweight pipeline (lowercase headers, remove
                        duplicates, conversions, binning, fillna, dropna,
                        encoding, outliers, scale), preview the effect, then
                        save a cleaned artifact. Databot follows along and
                        explains how the chosen steps impact the dataset.
                      </Section>
                      <Section title="Quick Note">
                        Data Cleaning and Feature Engineering is an iterative
                        process and involves many more functions. I realize that
                        the software here is not professionally complete. For
                        this to be truly useful, the component would have been
                        set up much differently and had many more steps and
                        options in the process.
                      </Section>

                      <Section title="How Databot gets context (frontend)">
                        Databot receives context in two ways:
                        <ol className="list-decimal list-inside mt-1 space-y-1">
                          <li>
                            A debounced POST to{" "}
                            <code>/api/databot/state/:id</code> carrying the
                            options, pipeline summary, and before/after stats.
                          </li>
                          <li>
                            Two window events Databot.jsx listens for:
                            <code>databot:context</code> (high-level plan and
                            results) and <code>dfjsx-cleaning-action</code> (the
                            last concrete action label).
                          </li>
                        </ol>
                      </Section>

                      <Section title="Frontend: push state + broadcast events">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={fePushState}
                        />
                      </Section>

                      <Section title="Frontend: record last action (for chat headers)">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feActionEvent}
                        />
                      </Section>

                      <Section title="Databot.jsx: consume events + prompt enrichment">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feDatabotGlue}
                        />
                      </Section>

                      <Section title="Frontend: stats delta & coach messages (UX hints)">
                        <CodeBlock language="javascript" dark code={feCoach} />
                      </Section>

                      <Section title="Backend: state endpoint (shape)">
                        <CodeBlock language="python" dark code={beStateApi} />
                      </Section>

                      <Section title="Optional UI: prime button">
                        <CodeBlock language="jsx" dark code={fePrimeButton} />
                      </Section>

                      <Section title="Notes" as="ul">
                        <li>
                          Keeping Databot’s context in sync across pages was
                          tricky—fixes on this screen sometimes broke others.
                          There are still a few things I want to change in this
                          databot. Next time I’ll split the logic into small,
                          page-scoped handlers with a shared, typed payload
                          instead of one large conditional.
                        </li>

                        <li>
                          <code>debouncedFetch</code> collapses rapid option
                          changes into a single update (~500 ms), preventing
                          “chat flapping” and redundant bot messages. A summary
                          gate also avoids sending no-op updates when the
                          pipeline hasn’t changed. For critical events (
                          <em>preview</em>, <em>preview_result</em>,{" "}
                          <em>save_request</em>, <em>save_success</em>,{" "}
                          <em>save_error</em>), that gate is bypassed so Databot
                          always sees the outcome, even if the summary text is
                          unchanged.
                        </li>

                        <li>
                          If users toggle the same plan repeatedly,
                          <code>lastPipelineSummary</code> can suppress pushes.
                        </li>
                        <li>
                          If headers are lowercased, ensure{" "}
                          <code>normalizeOpsForServer</code> and{" "}
                          <code>normalizePipelineForServer</code> are applied
                          before posting to <code>/clean-preview</code> so
                          column names match the server side.
                        </li>
                        <li>
                          Databot chat on <code>/cleaning</code> prepends{" "}
                          <em>Plan</em>, <em>Last action</em>, and{" "}
                          <em>Result</em> to the user’s question so responses
                          are specific to what just changed.
                        </li>
                      </Section>

                      <Section title="Takeaway">
                        The key to useful guidance here is{" "}
                        <strong>just-in-time context</strong>: push a compact
                        summary of the pipeline and the preview deltas, then
                        broadcast a lightweight event so Databot can talk about
                        the exact steps the user just tried. This takes alot of
                        trial and error testing and reordering of the code and
                        revision of prompts.
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
