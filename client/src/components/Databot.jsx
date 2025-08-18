import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import appInfo from "../data/databot_app_info.md?raw";
import CatBotSVG from "./CatBotSVG.jsx";

export default function Databot({ selectedDataset }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  const location = useLocation();
  const match = location.pathname.match(/\/datasets\/(\d+)/);
  const datasetId = match ? parseInt(match[1]) : selectedDataset || null;

  const API_BASE =
    import.meta.env.MODE === "development" ? "http://127.0.0.1:8000" : "";

  const [botType, setBotType] = useState("databot");
  const [forcedContext, setForcedContext] = useState(null);
  const [hasPrimed, setHasPrimed] = useState(false);
  const [cleaningContext, setCleaningContext] = useState({
    pipelineSummary: "",
    resultSummary: "",
    lastAction: "",
  });

  // Route flags
  const isAppInfoRoute =
    location.pathname === "/dashboard" || location.pathname === "/models";
  const isPredictors = location.pathname === "/predictors";

  const hasPredictorContext = Boolean(
    (forcedContext && (forcedContext.result || forcedContext.feature)) ||
      (selectedDataset && selectedDataset.latestPredictorResult)
  );

  const canChat = isAppInfoRoute
    ? true
    : isPredictors
    ? hasPredictorContext
    : true;

  useEffect(() => {
    const h = (e) => {
      setForcedContext(null);
      setBotType(e.detail?.botType || "databot");
      setForcedContext(e.detail?.context || null);
      setIsOpen(true);
      if (e.detail?.botType === "modelbot") {
        setInput(
          e.detail?.context?.feature?.startsWith?.("college_earnings")
            ? "Explain the college earnings results and suggest improvements."
            : "Explain the accessibility results and suggest a rewrite."
        );
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
  }, []);

  useEffect(() => {
    const onDatabotContext = (e) => {
      const d = e.detail || {};
      if (d.page !== "data-cleaning") return;

      setIsOpen(true);

      if (
        d.intent === "options_updated" &&
        (d.summary || d.selectedOpsSummary)
      ) {
        const plan = d.summary || d.selectedOpsSummary;
        setCleaningContext((prev) => ({
          ...prev,
          pipelineSummary: plan || "",
        }));
        const initial =
          typeof d.initial_message === "string" && d.initial_message.trim()
            ? d.initial_message.trim()
            : "I see you’re making changes to your dataset for preview. Would you like me to explain them?";

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: plan ? `${initial}\n\nPlan:\n${plan}` : initial,
          },
        ]);
      }

      if (d.intent === "preview_result" && d.result_summary) {
        setCleaningContext((prev) => ({
          ...prev,
          resultSummary: d.result_summary || "",
        }));
      }
    };

    window.addEventListener("databot:context", onDatabotContext);
    return () =>
      window.removeEventListener("databot:context", onDatabotContext);
  }, []);

  // Capture the last concrete action label (e.g., "Filled NA in 'Mass' with median")
  useEffect(() => {
    const onCleaningAction = (e) => {
      const a = e.detail?.action || "";
      if (a) {
        setIsOpen(true);
        setCleaningContext((prev) => ({ ...prev, lastAction: a }));
      }
    };
    window.addEventListener("dfjsx-cleaning-action", onCleaningAction);
    return () =>
      window.removeEventListener("dfjsx-cleaning-action", onCleaningAction);
  }, []);

  useEffect(() => {
    setHasPrimed(false);
  }, [botType, forcedContext?.feature, forcedContext?.version]);

  useEffect(() => {
    if (/\/datasets\/\d+/.test(location.pathname)) {
      setBotType("databot");
      setForcedContext(null);
      setHasPrimed(false);
    }
  }, [location.pathname]);

  useEffect(() => {
    if (location.pathname === "/dashboard" || location.pathname === "/models") {
      setBotType("databot");
      setForcedContext({ app_info_text: appInfo });
      setHasPrimed(false);
      setMessages([
        {
          role: "assistant",
          content:
            "Welcome to df.jsx! I’m Databot, here to guide you. Ask about the app’s features, workflow, or tips, or explore dataset details and predictor models!",
        },
      ]);
    } else {
      setMessages([]);
    }
  }, [location.pathname]);

  // --- Models page advisor helpers ---
  function detectTaskFromQuestion(q) {
    const s = (q || "").toLowerCase();
    const mentionsLogistic = /(logistic|sigmoid)/.test(s);
    const mentionsRF = /(random forest|rf|multiclass|multi-class)/.test(s);
    const mentionsCluster = /(cluster|clustering|kmeans|k-means|pca)/.test(s);
    const mentionsBinary = /\bbinary\b|\byes\b|\bno\b|\btrue\b|\bfalse\b/.test(
      s
    );

    // Disambiguation: both logistic + RF mentioned
    if (mentionsLogistic && mentionsRF) {
      return mentionsBinary ? "logistic" : "multiclass";
    }

    if (mentionsLogistic) return "logistic";
    if (mentionsRF) return "multiclass";
    if (mentionsCluster) return "cluster";
    return null;
  }

  async function fetchWithTimeout(url, options = {}, ms = 4000) {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), ms);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(t);
    }
  }

  async function fetchAdvisor(API_BASE, task) {
    const url = `${API_BASE}/api/databot/cleaned_datasets/recommendations?task=${encodeURIComponent(
      task
    )}`;
    const res = await fetchWithTimeout(
      url,
      { method: "GET", credentials: "include" },
      4000
    );
    if (!res.ok) throw new Error(`advisor ${res.status}`);
    return res.json();
  }

  function summarizeAdvisorResult(advisorJson) {
    const items = advisorJson?.items || [];
    if (!items.length) return "Advisor: No matching cleaned datasets found.";

    const lines = items.slice(0, 8).map((it) => {
      const badges = (it.badges || []).slice(0, 3).join(", ");
      const why = (it.why || [])[0] || "";
      const b = it.candidates?.binary?.slice?.(0, 2) || [];
      const m = it.candidates?.multiclass?.slice?.(0, 2) || [];
      const cand = b.length
        ? ` (binary: ${b.join(", ")})`
        : m.length
        ? ` (multiclass: ${m.join(", ")})`
        : "";

      return `• ${it.title}${badges ? ` — [${badges}]` : ""}${
        why ? ` — ${why}` : ""
      }${cand}`;
    });

    return [`Advisor results for task "${advisorJson.task}":`, ...lines].join(
      "\n"
    );
  }

  const askDatabot = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    if (isPredictors && !hasPredictorContext) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const effectiveBotType = isAppInfoRoute ? "databot" : botType;
      const ctx = isAppInfoRoute ? { app_info_text: appInfo } : forcedContext;

      let question = userMessage.content;
      let url = `${API_BASE}/api/databot/query`;
      let payload;

      // --------------------- BEGIN: Models page advisor branch ---------------------
      let usedAdvisor = false;
      if (location.pathname === "/models") {
        let task = detectTaskFromQuestion(question);

        // (A) If we recognized a task (logistic/multiclass/cluster), keep your existing behavior
        if (task) {
          try {
            const advisor = await fetchAdvisor(API_BASE, task);
            const summary = summarizeAdvisorResult(advisor);

            url = `${API_BASE}/api/databot/query_welcome`;
            question = `${summary}
      
      System: From these advisor results, recommend the single best dataset and, if supervised, the exact target column to use. Then give 3 setup steps (cleaning/encoding/validation) specific to that dataset. Be concise (<120 words).
      Restrict your answer to datasets listed above (these are the ones visible on the Models page).
      
      User question: ${userMessage.content}`;

            payload = { question, app_info: appInfo };
            usedAdvisor = true;
          } catch (e) {
            url = `${API_BASE}/api/databot/query_welcome`;
            question = `User asked about dataset-model fit on the Models page.\n(Advisor failed: ${
              e?.message || e
            })\n\nQuestion: ${userMessage.content}`;
            payload = { question, app_info: appInfo };
            usedAdvisor = true;
          }
        }

        // (B) If NO task was detected (e.g., “tell me about Shopping Trends”),
        // still fetch a dataset snapshot so answers stay page-specific.
        if (!task && !usedAdvisor) {
          try {
            // Use a neutral task (cluster) just to get the current page’s dataset snapshot
            const advisor = await fetchAdvisor(API_BASE, "cluster");
            const snapshot = summarizeAdvisorResult(advisor);

            url = `${API_BASE}/api/databot/query_welcome`;
            question = `${snapshot}
      
      System: Use ONLY the datasets listed above (these reflect what's visible on the Models page right now).
      If the user asks about a dataset not in the list, say you don’t see it on this page and suggest the closest match.
      Keep answers ≤120 words.
      
      User question: ${userMessage.content}`;

            payload = { question, app_info: appInfo };
            usedAdvisor = true;
          } catch (e) {
            // fall through; the generic app-info branch will handle it below
          }
        }
      }

      // ---------------------- Models page advisor branch ----------------------

      if (!usedAdvisor) {
        if (isAppInfoRoute && ctx?.app_info_text) {
          url = `${API_BASE}/api/databot/query_welcome`;
          question = `App Info:\n${ctx.app_info_text}\n\nQuestion: ${question}`;
          payload = { question, app_info: ctx.app_info_text };
        } else if (isPredictors && botType === "modelbot" && ctx) {
          let header = "";
          if (ctx?.feature?.startsWith?.("college_earnings")) {
            const drivers =
              Array.isArray(ctx?.result?.drivers) && ctx.result.drivers.length
                ? `Drivers: ${ctx.result.drivers
                    .map((d) => `${d.direction}${d.factor}`)
                    .join(", ")}`
                : null;
            const pct =
              typeof ctx?.result?.prob === "number"
                ? Math.round(ctx.result.prob * 100)
                : null;
            header = [
              "Context: College Earnings — 5y ≥ $75k.",
              `Inputs: CIP4=${ctx?.inputs?.cip4 || "?"}, Degree=${
                ctx?.inputs?.degree_level || "?"
              }, State=${ctx?.inputs?.state || "?"}${
                ctx?.inputs?.public_private
                  ? `, Type=${ctx.inputs.public_private}`
                  : ""
              }`,
              pct != null && ctx?.result?.bucket
                ? `Score: ${ctx.result.bucket} (${pct}%).`
                : null,
              drivers,
            ]
              .filter(Boolean)
              .join("\n");
          } else if (ctx?.feature === "accessibility_risk") {
            const confusion =
              Array.isArray(ctx?.result?.confusion_sources) &&
              ctx.result.confusion_sources.length
                ? `Confusion: ${ctx.result.confusion_sources
                    .map((s) => `${s.type}: ${s.evidence?.join(", ")}`)
                    .join(" | ")}`
                : null;
            const pct =
              typeof ctx?.result?.prob === "number"
                ? Math.round(ctx.result.prob * 100)
                : null;
            header = [
              "Context: Accessibility Misinterpretation Risk.",
              `Audience: ${ctx?.inputs?.audience || "?"}, Medium: ${
                ctx?.inputs?.medium || "?"
              }, Intent: ${ctx?.inputs?.intent ?? "—"}`,
              pct != null && ctx?.result?.bucket
                ? `Score: ${ctx.result.bucket} (${pct}%).`
                : null,
              confusion,
              ctx?.result?.rewrite
                ? `Rewrite ≤15: ${ctx.result.rewrite}`
                : null,
            ]
              .filter(Boolean)
              .join("\n");
          } else {
            header = "Context: Unknown model.";
          }
          question = `${header}\n\nUser: ${question}`;
          payload = {
            question,
            bot_type: effectiveBotType,
            model_context: ctx,
          };
        } else if (location.pathname.includes("/cleaning")) {
          // Prepend exact cleaning context so the backend can explain what *just* changed
          const headerParts = [];
          if (cleaningContext?.pipelineSummary)
            headerParts.push(`Plan:\n${cleaningContext.pipelineSummary}`);
          if (cleaningContext?.lastAction)
            headerParts.push(`Last action:\n${cleaningContext.lastAction}`);
          if (cleaningContext?.resultSummary)
            headerParts.push(`Result:\n${cleaningContext.resultSummary}`);
          const header = headerParts.length
            ? headerParts.join("\n\n") + "\n\n"
            : "";

          const directive = `System: You are Databot on the Data Cleaning page.
            Explain the implications of the latest cleaning changes on this dataset using the context below.
            Be concrete and concise (≤120 words). Focus on effects on row count, nulls, dtypes, scaling/encoding, and downstream modeling impact.
            
            Context:
            ${header || "(no recent context)"}
            Dataset ID: ${datasetId ?? "unknown"}`;

          question = `${directive}\n\nUser: ${userMessage.content}`;

          payload = {
            question,
            bot_type: effectiveBotType,
            ...(datasetId != null ? { dataset_id: datasetId } : {}),
            app_info: appInfo,
          };
        } else {
          payload = {
            question,
            bot_type: effectiveBotType,
            ...(datasetId != null ? { dataset_id: datasetId } : {}),
            app_info: appInfo,
          };
        }
      }

      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      let data;
      try {
        data = await res.json();
      } catch {
        data = { detail: await res.text() };
      }

      const assistantMessage = res.ok
        ? {
            role: "assistant",
            content: data.answer || data.message || "(no answer)",
          }
        : {
            role: "assistant",
            content: isAppInfoRoute
              ? "I’m here to help with df.jsx. Ask about features or workflow!"
              : botType === "modelbot"
              ? "I can’t process that right now. Ask about the model results!"
              : "Error: Couldn’t fetch a response. Try again or check your dataset!",
          };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const isAppInfoRouteFallback =
        location.pathname === "/dashboard" || location.pathname === "/models";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: isAppInfoRouteFallback
            ? "I’m here to help with df.jsx. Ask about features or workflow!"
            : botType === "modelbot"
            ? "I can’t process that right now. Ask about the model results!"
            : "Error: Couldn’t fetch a response. Try again or check your dataset!",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="fixed bottom-4 right-4 w-96 z-50">
      {/* CLOSED: floating CatBot toggle */}
      {!isOpen && (
        <div
          onClick={() => setIsOpen(true)}
          className="ml-auto rounded-full shadow-lg ring-1 ring-black/5 bg-white p-2 cursor-pointer hover:scale-[1.03] transition db-bob"
          title="Open Databot"
          aria-label="Open Databot"
        >
          <CatBotSVG className="db-blink" />
        </div>
      )}

      {/* OPEN: panel with perched mascot */}
      {isOpen && (
        <div className="relative mt-8">
          {/* Perched, larger mascot overlapping the top edge */}
          <div className={`db-perch ${isLoading ? "db-tilt" : ""}`}>
            <CatBotSVG
              size={88}
              stroke="#233a88"
              accent="#16b9a6"
              fill="#ffffff"
              className="db-blink"
            />
          </div>

          {/* Actual dialog panel */}
          <div
            className={`bg-white border rounded-2xl shadow-xl overflow-hidden flex flex-col h-[500px] ${
              isLoading ? "db-glow" : ""
            }`}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b bg-cyan-300 text-white">
              <h2 className="text-lg font-bold tracking-wide">Databot Tutor</h2>
              <button
                onClick={() => setIsOpen(false)}
                className="rounded-md px-2 py-1 text-white/90 hover:text-white hover:bg-white/10"
                title="Close"
                aria-label="Close Databot"
              >
                ✕
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-slate-50">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex items-start ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center mr-2 ring-1 ring-black/5">
                      <CatBotSVG
                        size={20}
                        stroke="#233a88"
                        accent="#16b9a6"
                        fill="#ffffff"
                      />
                    </div>
                  )}
                  <div
                    className={`px-3 py-2 rounded-2xl max-w-[75%] leading-relaxed shadow-sm ${
                      msg.role === "user"
                        ? "bg-indigo-100 text-slate-900 ml-auto rounded-br-md"
                        : "bg-cyan-100 text-slate-900 rounded-bl-md"
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="text-gray-500 italic flex items-center gap-2">
                  <span className="db-typing inline-flex gap-1">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                  Databot is thinking…
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            {isPredictors && !canChat && (
              <div className="flex items-center gap-2 px-3 py-2 mx-3 mb-2 rounded-md border border-blue-300 bg-blue-50 text-sm text-blue-700">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 text-blue-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                <span>
                  <span className="font-medium">Databot is standing by.</span>{" "}
                  Run any predictor and I’ll explain the results and next steps.
                </span>
              </div>
            )}

            {/* Input */}
            <form onSubmit={askDatabot} className="p-3 border-t flex">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  !canChat && isPredictors
                    ? "Run a predictor to enable Databot"
                    : isAppInfoRoute
                    ? "Ask about this app…"
                    : botType === "modelbot"
                    ? "Ask about this prediction..."
                    : "Ask about your dataset..."
                }
                className={`flex-grow border rounded-l px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400 ${
                  !canChat && isPredictors
                    ? "bg-gray-100 text-gray-500 cursor-not-allowed"
                    : ""
                }`}
                disabled={isLoading || (!canChat && isPredictors)}
              />
              <button
                type="submit"
                className={`px-4 rounded-r text-white ${
                  !canChat && isPredictors
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-orange-500 hover:bg-emerald-500"
                }`}
                disabled={isLoading || (!canChat && isPredictors)}
              >
                Send
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
