import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import logo from "../assets/newlogo500.png"; // ✅ import your logo

export default function Databot({ selectedDataset }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Detect datasetId from the URL (e.g., /datasets/22)
  const location = useLocation();
  const match = location.pathname.match(/\/datasets\/(\d+)/);
  const datasetId = match ? parseInt(match[1]) : selectedDataset || null;

  const API_BASE =
    import.meta.env.MODE === "development" ? "http://127.0.0.1:8000" : "";

  // NEW: allow opening as "ModelBot" with prediction context
  const [botType, setBotType] = useState("databot"); // "databot" | "modelbot"
  const [forcedContext, setForcedContext] = useState(null);
  const [hasPrimed, setHasPrimed] = useState(false); // prepend context once

  // Open + set context (explicit open)
  useEffect(() => {
    const h = (e) => {
      setBotType(e.detail?.botType || "databot");
      setForcedContext(e.detail?.context || null);
      setIsOpen(true);
      if (e.detail?.botType === "modelbot") {
        setInput("Explain these results and suggest a clearer rewrite.");
      }
    };
    window.addEventListener("dfjsx-open-bot", h);
    return () => window.removeEventListener("dfjsx-open-bot", h);
  }, []);

  // Remember model context after a model run (no auto-open)
  useEffect(() => {
    const setCtx = (e) => {
      setBotType(e.detail?.botType || "databot");
      setForcedContext(e.detail?.context || null);
    };
    window.addEventListener("dfjsx-set-bot-context", setCtx);
    return () => window.removeEventListener("dfjsx-set-bot-context", setCtx);
  }, []);

  // If mode or context changes, allow header to be prepended once again
  useEffect(() => {
    setHasPrimed(false);
  }, [botType, forcedContext?.feature, forcedContext?.version]);

  // If we navigate to a dataset detail page, default back to dataset mode
  useEffect(() => {
    if (/\/datasets\/\d+/.test(location.pathname)) {
      setBotType("databot");
      setForcedContext(null);
      setHasPrimed(false);
    }
  }, [location.pathname]);

  // Passive listener: remember model context after a model run (no auto-open)
  useEffect(() => {
    const setCtx = (e) => {
      setBotType(e.detail?.botType || "databot");
      setForcedContext(e.detail?.context || null);
    };
    window.addEventListener("dfjsx-set-bot-context", setCtx);
    return () => window.removeEventListener("dfjsx-set-bot-context", setCtx);
  }, []);

  // If mode or context changes, allow header to be prepended once again
  useEffect(() => {
    setHasPrimed(false);
  }, [botType, forcedContext?.feature, forcedContext?.version]);

  // Store model context without opening (silent set)
  useEffect(() => {
    const setCtx = (e) => {
      setBotType(e.detail?.botType || "databot");
      setForcedContext(e.detail?.context || null);
    };
    window.addEventListener("dfjsx-set-bot-context", setCtx);
    return () => window.removeEventListener("dfjsx-set-bot-context", setCtx);
  }, []);

  // Reset the one-time header when mode/context changes
  useEffect(() => {
    setHasPrimed(false);
  }, [botType, forcedContext?.feature, forcedContext?.version]);

  const askDatabot = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Build question; always include model_context in model mode.
      // Prepend a readable header only on the first message.
      // Always include model_context in model mode; prepend readable header only once
      // Always include model_context in model mode; prepend readable header only once
      let question = userMessage.content;
      const ctx = forcedContext;
      const isModel = botType === "modelbot" && !!ctx;
      const model_context = isModel ? ctx : null;

      if (isModel && !hasPrimed) {
        const pct =
          typeof ctx?.result?.prob === "number"
            ? Math.round(ctx.result.prob * 100)
            : null;

        let header = "";
        if (ctx?.feature?.startsWith?.("college_earnings")) {
          const drivers =
            Array.isArray(ctx?.result?.drivers) && ctx.result.drivers.length
              ? `Drivers: ${ctx.result.drivers
                  .map((d) => `${d.direction}${d.factor}`)
                  .join(", ")}`
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
        } else {
          const confusion =
            Array.isArray(ctx?.result?.confusion_sources) &&
            ctx.result.confusion_sources.length
              ? `Confusion: ${ctx.result.confusion_sources
                  .map((s) => `${s.type}: ${s.evidence?.join(", ")}`)
                  .join(" | ")}`
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
            ctx?.result?.rewrite ? `Rewrite ≤15: ${ctx.result.rewrite}` : null,
          ]
            .filter(Boolean)
            .join("\n");
        }

        question = `${header}\n\nUser: ${question}`;
        setHasPrimed(true);
      }

      // --- Request: support both dataset and prediction use-cases ---
      let url = `${API_BASE}/api/databot/query`;

      const basePayload = {
        question,
        bot_type: botType,
        ...(model_context ? { model_context } : {}),
      };

      const payload =
        datasetId != null
          ? { ...basePayload, dataset_id: datasetId }
          : basePayload;

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
            content: "Error: " + (data.detail || `HTTP ${res.status}`),
          };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: " + err.message },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="fixed bottom-4 right-4 w-96 z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-blue-600 text-white px-4 py-2 rounded shadow-lg hover:bg-blue-500"
      >
        {isOpen ? "Close Databot" : "💬 Open Databot"}
      </button>

      {isOpen && (
        <div className="mt-2 bg-white border rounded-lg shadow-lg flex flex-col h-[500px]">
          <h2 className="text-lg font-bold p-3 border-b">Databot Tutor</h2>

          {/* Conversation window */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex items-start ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {msg.role === "assistant" && (
                  <img
                    src={logo}
                    alt="Databot Logo"
                    className="w-8 h-8 rounded-full mr-2"
                  />
                )}
                <div
                  className={`p-2 rounded-lg max-w-[75%] ${
                    msg.role === "user"
                      ? "bg-blue-100 text-right ml-auto"
                      : "bg-green-100"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="text-gray-500 italic">Databot is thinking...</div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input form */}
          <form onSubmit={askDatabot} className="p-3 border-t flex">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                botType === "modelbot"
                  ? "Ask about this prediction..."
                  : "Ask about your dataset..."
              }
              className="flex-grow border rounded-l px-3 py-2"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="bg-green-600 text-white px-4 rounded-r hover:bg-green-500"
              disabled={isLoading}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
