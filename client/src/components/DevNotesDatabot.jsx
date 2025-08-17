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
   * SNIPPETS
   * ========================= */

  // A) Request/Response contract (what the page sends/receives)
  const exRequest = `{
  "page": "datasets/detail",
  "user_message": "Is this dataset okay for logistic regression?",
  "dataset_id": 42,
  "context_hints": ["model_selection", "data_quality"]
}`;

  const exResponse = `{
  "answer": "Logistic regression expects a binary target. Your 'churned' is 0/1...",
  "used_context": {
    "page": "datasets/detail",
    "dataset_id": 42,
    "columns": ["age","income","churned"],
    "n_rows": 12134,
    "has_missing_values": true
  },
  "model": "gpt-*",
  "prompt_revision": "v3",
  "latency_ms": 612,
  "input_tokens": 962,
  "output_tokens": 164,
  "request_id": "dbot_2025-08-16T18:05:11Z_7f2c"
}`;

  // B) FastAPI route: schema-first ask endpoint
  const beRouteAsk = `# server/routers/databot.py (excerpt)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from .deps import get_db
from .llm import call_llm
from .context import build_context
from .telemetry import with_request_id

router = APIRouter(prefix="/api/databot", tags=["databot"])

class AskRequest(BaseModel):
    page: str = Field(..., examples=["datasets/detail", "models/index"])
    user_message: str
    dataset_id: int | None = None
    context_hints: list[str] | None = None

class AskResponse(BaseModel):
    answer: str
    used_context: dict
    model: str
    prompt_revision: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    request_id: str

@router.post("/ask", response_model=AskResponse)
@with_request_id
async def ask(req: AskRequest, db: AsyncSession = Depends(get_db)) -> AskResponse:
    # 1) Build page-aware context (optionally queries MySQL)
    ctx = await build_context(db=db, page=req.page, dataset_id=req.dataset_id,
                             hints=req.context_hints or [])
    # 2) Render prompt (versioned per page)
    prompt, rev = render_prompt(page=req.page, user=req.user_message, ctx=ctx)

    # 3) Model call with guardrails
    llm_out = await call_llm(prompt=prompt, temperature=0.2, max_tokens=600, timeout_s=20)
    if not llm_out.ok:
        raise HTTPException(status_code=502, detail=f"LLM error: {llm_out.reason}")

    # 4) Typed response with telemetry
    return AskResponse(
        answer=llm_out.text,
        used_context=ctx.public_dict(),
        model=llm_out.model,
        prompt_revision=rev,
        latency_ms=llm_out.latency_ms,
        input_tokens=llm_out.usage.prompt_tokens,
        output_tokens=llm_out.usage.completion_tokens,
        request_id=llm_out.request_id
    )`;

  // C) Context builder (MySQL -> prompt inputs)
  const beContext = `# server/routers/context.py (excerpt)
from types import SimpleNamespace
from sqlalchemy import select
from .models import Dataset

async def build_context(db, page: str, dataset_id: int | None, hints: list[str]):
    ctx = {"page": page, "hints": hints}

    if dataset_id is not None:
        row = (await db.execute(select(Dataset).where(Dataset.id == dataset_id))).scalar_one_or_none()
        if row:
            ctx |= {
                "dataset_id": row.id,
                "columns": row.column_metadata.get("names", []),
                "target": row.target_column,
                "n_rows": row.n_rows,
                "n_columns": row.n_columns,
                "has_missing_values": row.has_missing_values,
            }
    return SimpleNamespace(
        **ctx,
        public_dict=lambda: ctx  # safely exposes only public bits
    )`;

  // D) Prompt template (page-aware, versioned)
  const bePrompt = `# server/routers/prompts.py (excerpt)
PROMPT_VERSIONS = {
    "datasets/detail": "v3",
    "models/index": "v2",
}

def render_prompt(page: str, user: str, ctx) -> tuple[str, str]:
    rev = PROMPT_VERSIONS.get(page, "v1")
    if page == "datasets/detail":
        tmpl = f"""
You are a data science tutor. Answer briefly and concretely.
Context:
- dataset_id: {getattr(ctx, 'dataset_id', None)}
- columns: {getattr(ctx, 'columns', [])}
- target: {getattr(ctx, 'target', None)}
- n_rows: {getattr(ctx, 'n_rows', 'unknown')}
- missing_values: {getattr(ctx, 'has_missing_values', False)}
User question: "{user}"
Rules:
- Prefer guidance tied to the dataset context above.
- If data is insufficient, state what is missing and suggest the next concrete step.
""".strip()
    else:
        tmpl = f"You are a data science tutor. Page: {page}. Hints: {ctx.hints}\\nUser question: \\"{user}\\"\\nKeep answers actionable and tied to the page."
    return tmpl, rev`;

  // E) Minimal frontend call (React) — no template literals to keep it copy-safe here
  const feAskCall = `// client/src/components/Databot.jsx (excerpt)
async function askDatabot(userMessage, location) {
  const match = (location.pathname || "").match(/\\/datasets\\/(\\d+)/);
  const datasetId = match ? Number(match[1]) : null;

  const payload = {
    page: routeToPageKey(location.pathname),
    user_message: userMessage,
    dataset_id: datasetId,
    context_hints: ["model_selection", "data_quality"]
  };

  const res = await fetch("/api/databot/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload)
  });

  if (!res.ok) throw new Error("Databot failed: " + res.status);
  return await res.json();
}`;

  return (
    <>
      {/* Floating button (bottom-left like other pages) */}
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
                        Dev Notes - Databot
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
                        In class we built prompts in notebooks and used a simple
                        hosted runtime. Here, every Databot question is a real
                        API request that: (1) gathers page-specific context, (2)
                        optionally pulls dataset metadata from MySQL, (3)
                        renders a versioned prompt with guardrails, (4) returns
                        a typed response to the UI with telemetry.
                      </Section>

                      <Section title=" Basic Architecture" as="ul">
                        <li>
                          <strong>Frontend (React):</strong> sends{" "}
                          <code>page</code>, optional <code>dataset_id</code>,
                          and the user’s question to{" "}
                          <code>/api/databot/ask</code>.
                        </li>
                        <li>
                          <strong>Backend (FastAPI):</strong> validates with
                          Pydantic, loads MySQL metadata when needed, renders a{" "}
                          <em>page-aware</em> prompt, calls the LLM, and returns
                          a typed response.
                        </li>
                        <li>
                          <strong>MySQL (SQLAlchemy):</strong> stores dataset
                          facts (columns, target, n_rows, missingness) so
                          answers are grounded.
                        </li>
                        <li>
                          <strong>Observability:</strong> latency, token usage,
                          and a <code>request_id</code> are captured per call.
                        </li>
                      </Section>
                      <Section title="Varied Databot Context Sources">
                        <ul className="list-disc list-inside space-y-1">
                          <li>
                            <strong>Dashboard:</strong> Reads the project
                            overview from the local markdown and injects that
                            text as read-only reference material. No dataset
                            lookup here.
                          </li>

                          <li>
                            <strong>Dataset Detail:</strong> Extracts{" "}
                            <code>dataset_id</code> from the URL and the backend
                            loads metadata from MySQL via SQLAlchemy (columns,
                            target, row/column counts, missingness, etc.). That
                            dictionary is merged into the prompt so answers are
                            grounded in the selected dataset.
                          </li>

                          <li>
                            <strong>Data Cleaning:</strong> Same{" "}
                            <code>dataset_id</code> metadata as Dataset Detail,
                            plus (when available) the cleaning
                            preview/suggestions returned by the cleaning API
                            (e.g., dtype map, null counts, candidate
                            imputations/encodings). If the preview hasn’t been
                            run, it falls back to metadata-only.
                          </li>

                          <li>
                            <strong>Models:</strong> Aggregates metadata for{" "}
                            <em>all</em> cleaned datasets shown on the page via
                            a small advisor endpoint (badges, target candidates,
                            signals), merges in model descriptors
                            (hints/constraints for Logistic, RF, PCA+KMeans,
                            etc.), and app overview text from the same markdown
                            used on the Dashboard. When the user asks a
                            model-selection question, the advisor summary is
                            prepended to the prompt so the model can recommend a
                            dataset + target with setup steps.
                          </li>

                          <li>
                            <strong>Predictors:</strong> Context is{" "}
                            <em>result-driven</em>. After a predictor run
                            completes, the user clicks “Explain with Databot,”
                            which packages the predictor type + params, key
                            outputs/metrics, and (if present) the{" "}
                            <code>dataset_id</code>
                            metadata. That bundle is sent to Databot to generate
                            a plain-English explanation of the results and
                            recommended next actions.
                          </li>
                        </ul>
                      </Section>

                      <Section title="Request / Response contract">
                        <div className="grid gap-3">
                          <CodeBlock language="json" dark code={exRequest} />
                          <CodeBlock language="json" dark code={exResponse} />
                        </div>
                      </Section>

                      <Section title="FastAPI route: schema-first design">
                        <CodeBlock language="python" dark code={beRouteAsk} />
                      </Section>

                      <Section title="Context builder: MySQL → prompt inputs">
                        <CodeBlock language="python" dark code={beContext} />
                      </Section>

                      <Section title="Prompt template: page-aware & versioned">
                        <CodeBlock language="python" dark code={bePrompt} />
                      </Section>

                      <Section title="Frontend call (React)">
                        <CodeBlock
                          language="javascript"
                          dark
                          code={feAskCall}
                        />
                      </Section>

                      <Section title="Chatbot in Deployed Environment" as="ul">
                        <li>
                          <strong>Schema-driven:</strong> requests and responses
                          are typed (Pydantic), making failures reproducible.
                        </li>
                        <li>
                          <strong>Grounded answers:</strong> prompts incorporate
                          dataset metadata when a dataset is in view.
                        </li>
                        <li>
                          <strong>Separation of concerns:</strong> context,
                          prompt, and model call are modular and versioned.
                        </li>
                        <li>
                          <strong>Guardrails:</strong> low temperature, timeouts
                          and token caps keep answers concise and on-task.
                        </li>
                        <li>
                          <strong>Versioning:</strong>{" "}
                          <code>prompt_revision</code> enables A/B comparisons.
                        </li>
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
