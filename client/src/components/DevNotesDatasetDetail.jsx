// client/src/components/DevNotesDatasetDetail.jsx
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

export default function DevNotesDatasetDetail() {
  const [open, setOpen] = useState(false);

  /* =========================
   * SNIPPETS (what I actually changed / rely on)
   * ========================= */

  // A) DatasetDetail: fetch detail + optional cleaned preview
  const feFetchDataset = `// DatasetDetail.jsx (excerpt)
useEffect(() => {
  const fetchDataset = async () => {
    const res = await fetch(\`/api/datasets/\${id}\`, { credentials: "include" });
    if (res.status === 401) return navigate("/login");
    if (res.status === 404) return navigate("/datasets");
    if (!res.ok) throw new Error(\`Error \${res.status}\`);
    const data = await res.json();
    setDataset(data);

    // If cleaned artifact exists, load a small preview for the right panel
    if (data.s3_key_cleaned) {
      const cleanRes = await fetch(\`/api/datasets/\${id}/insights?which=cleaned\`, { credentials: "include" });
      const cleanData = await cleanRes.json();
      setCleanedPreview(cleanData.preview || []);
    }
  };
  fetchDataset();
}, [id]);`;

  // B) Prime Databot from DatasetDetail (exact context payload)
  const fePrimeDatabot = `// DatasetDetail.jsx (helper inside component)
function primeDatabotFromDetail(dataset, id) {
  const cm = dataset?.column_metadata || {};
  const topNulls = Object.entries(cm)
    .map(([col, meta]) => ({ col, nulls: Number(meta?.null_count ?? 0) }))
    .sort((a, b) => b.nulls - a.nulls)
    .slice(0, 6);
  const dtypeCounts = Object.entries(cm).reduce((acc, [_, meta]) => {
    const dt = String(meta?.dtype ?? "unknown");
    acc[dt] = (acc[dt] || 0) + 1;
    return acc;
  }, {});
  const context = {
    page: "dataset-detail",
    datasetId: Number(id),
    meta: {
      title: dataset?.title,
      n_rows: dataset?.n_rows,
      n_columns: dataset?.n_columns,
      has_missing_values: !!dataset?.has_missing_values,
      target_column: dataset?.target_column ?? null,
      has_cleaned: !!dataset?.s3_key_cleaned,
      dtypes_count: dtypeCounts,
      top_nulls: topNulls,
    },
  };

  // 1) Set context so Databot tailors next answer
  window.dispatchEvent(new CustomEvent("dfjsx-set-bot-context", {
    detail: { botType: "databot", context }
  }));
  // 2) Optionally open chat immediately
  window.dispatchEvent(new CustomEvent("dfjsx-open-bot", {
    detail: { botType: "databot", context }
  }));
}`;

  // C) Databot.jsx: listeners that make this work (already present in your file)
  const feDatabotListeners = `// Databot.jsx (already in your version)
useEffect(() => {
  const h = (e) => {
    setForcedContext(null);
    setBotType(e.detail?.botType || "databot");
    setForcedContext(e.detail?.context || null);
    setIsOpen(true);
    if (e.detail?.botType === "modelbot") {
      setInput(...); // seed question for model bot
    }
  };
  window.addEventListener("dfjsx-open-bot", h);
  return () => window.removeEventListener("dfjsx-open-bot", h);
}, []);

useEffect(() => {
  const setCtx = (e) => {
    setBotType(e.detail?.botType || "databot");
    setForcedContext(e.detail?.context || null);
  };
  window.addEventListener("dfjsx-set-bot-context", setCtx);
  return () => window.removeEventListener("dfjsx-set-bot-context", setCtx);
}, []);`;

  // D) Event shape Databot expects for dataset-detail (so answers reference THIS dataset)
  const eventShape = `// Context shape sent via dfjsx-set-bot-context / dfjsx-open-bot
{
  "botType": "databot",
  "context": {
    "page": "dataset-detail",
    "datasetId": 59,
    "meta": {
      "title": "Shopping Trends",
      "n_rows": 8277,
      "n_columns": 50,
      "has_missing_values": true,
      "target_column": null,
      "has_cleaned": true,
      "dtypes_count": { "float64": 22, "int64": 5, "object": 23 },
      "top_nulls": [
        { "col": "Price", "nulls": 4529 },
        { "col": "Estimated Battery Life", "nulls": 6602 }
      ]
    }
  }
}`;

  // E) Contrast: cleaning page pushes live pipeline via 'databot:context'
  const feCleaningEvents = `// DataCleaning.jsx → when user edits options / previews:
window.dispatchEvent(new CustomEvent("databot:context", {
  detail: {
    page: "data-cleaning",
    dataset_id: Number(id),
    intent: "options_updated" | "preview_result",
    summary: "...human-readable pipeline plan...",
    result_summary: "...delta rows/cols, filled counts...",
    selectedOpsSummary: "...selected ops text...",
  }
}));

// Databot.jsx consumes this to prepend a "Plan / Last action / Result" header to the user question.`;

  // F) Optional UI: tiny “Prime Databot” button near your dataset header
  const fePrimeButton = `// In DatasetDetail.jsx header actions (example)
<button
  onClick={() => primeDatabotFromDetail(dataset, id)}
  className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-sm text-white/90 ring-1 ring-white/20 hover:bg-white/15"
  title="Prime Databot with this dataset"
>
  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3a9 9 0 100 18 9 9 0 000-18zm1 9h3l-4 4-4-4h3V8h2v4z"/></svg>
  Ask Databot about this dataset
</button>`;

  // G) Backend note: dataset detail returns preview + column_metadata (used above)
  const beDatasetDetail = `# server/routers/datasets.py (excerpt)
@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: int, db: AsyncSession = Depends(get_async_db), user=Depends(auth_user)):
    ds = await get_dataset(db, dataset_id, user_id=user.id)
    # ds includes: title, description, n_rows, n_columns, has_missing_values,
    #              target_column, s3_key_cleaned, column_metadata, preview_data
    return ds`;

  return (
    <>
      {/* Floating button (bottom-left) */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 left-6 z-[60] rounded-full bg-emerald-600 p-3 shadow-lg hover:bg-emerald-500 text-white"
        title="Dataset Detail — Dev Notes"
        aria-label="Open Dev Notes"
      >
        <InformationCircleIcon className="h-6 w-6" />
      </button>

      {/* Slide-over */}
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
                        Dataset Detail — Dev Notes
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
                        The <strong>Dataset Detail</strong> page loads a single
                        dataset (raw preview and optional cleaned preview if it
                        exists), shows quick stats, triggers heatmap/insights
                        endpoints, and provides actions (download, preprocess).
                        Here I also <strong>prime Databot</strong> with the
                        dataset’s metadata so answers are specific to what the
                        user is looking at.
                      </Section>

                      <Section title="How Databot gets context" as="ol">
                        <li>
                          Build a compact <em>dataset-detail</em> context (id +
                          quick stats) from <code>column_metadata</code>.
                        </li>
                        <li>
                          Dispatch{" "}
                          <code className="px-1 rounded bg-gray-100">
                            dfjsx-set-bot-context
                          </code>{" "}
                          so the next chat reply is grounded on this dataset.
                        </li>
                        <li>
                          Optionally I could dispatch{" "}
                          <code className="px-1 rounded bg-gray-100">
                            dfjsx-open-bot
                          </code>{" "}
                          to pop the chat immediately.
                        </li>
                        <li>
                          On the Cleaning page, richer live updates flow via{" "}
                          <code className="px-1 rounded bg-gray-100">
                            databot:context
                          </code>{" "}
                          (plan / last action / result). Decided to keep Dataset
                          Detail context simpler on this page.
                        </li>
                      </Section>

                      <Section title="Frontend: fetch dataset">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feFetchDataset}
                        />
                      </Section>

                      <Section title="Frontend: prime Databot from DatasetDetail">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={fePrimeDatabot}
                        />
                      </Section>

                      <Section title="Databot listeners (already in Databot.jsx)">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feDatabotListeners}
                        />
                      </Section>

                      <Section title="Event payload shape">
                        <CodeBlock language="json" dark code={eventShape} />
                      </Section>

                      <Section title="Cleaning page events (for contrast)">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feCleaningEvents}
                        />
                      </Section>

                      <Section title="Optional UI: a small button to prime Databot">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={fePrimeButton}
                        />
                      </Section>

                      <Section title="Backend note (dataset detail payload)">
                        <CodeBlock
                          language="python"
                          dark
                          code={beDatasetDetail}
                        />
                      </Section>

                      <Section title="Troubleshooting">
                        <ul className="list-disc list-inside space-y-1">
                          <li>
                            When answers were generic, had to make the{" "}
                            <code>dfjsx-set-bot-context</code> fire with a valid{" "}
                            <code>datasetId</code> and <code>meta</code>.
                          </li>
                          <li>
                            If chat doesn’t open, check for any errors thrown by{" "}
                            <code>dfjsx-open-bot</code> listener in
                            <code>Databot.jsx</code>.
                          </li>
                          <li>
                            On older datasets (uploaded before this change),{" "}
                            <code>column_metadata</code> may be
                            incomplete—earlier imports didn’t persist fields
                            like <code>dtype</code>, <code>n_unique</code>, or{" "}
                            <code>null_count</code> for every column. When that
                            happens, Databot/advisor can’t compute badges or
                            target candidates and will fall back to generic
                            tips. So I had to re-run a cleaning preview (or
                            re-upload) to refresh metadata. Saving the cleaned
                            result backfills these stats for future sessions.
                            These are things I learned while testing and an
                            example of why testing took so long.
                          </li>
                        </ul>
                      </Section>
                    </div>
                    {/* /Body */}
                  </Dialog.Panel>
                </Transition.Child>
              </div>
              {/* /right sheet container */}
            </div>
          </div>
        </Dialog>
      </Transition.Root>
    </>
  );
}
