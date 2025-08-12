import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import logo from "../assets/newlogo500.png";
import appInfo from "../data/databot_app_info.md?raw";

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

  const askDatabot = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const isAppInfoRoute =
        location.pathname === "/dashboard" || location.pathname === "/models";
      const isPredictors = location.pathname === "/predictors";
      const effectiveBotType = isAppInfoRoute ? "databot" : botType;
      const ctx = isAppInfoRoute ? { app_info_text: appInfo } : forcedContext;

      let question = userMessage.content;
      let url = `${API_BASE}/api/databot/query`;
      let payload;

      if (isAppInfoRoute && ctx?.app_info_text) {
        url = `${API_BASE}/api/databot/query_welcome`;
        question = `App Info:\n${ctx.app_info_text}\n\nQuestion: ${question}`;
        payload = {
          question,
          app_info: ctx.app_info_text,
        };
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
            ctx?.result?.rewrite ? `Rewrite ≤15: ${ctx.result.rewrite}` : null,
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
      } else {
        payload = {
          question,
          bot_type: effectiveBotType,
          ...(datasetId != null ? { dataset_id: datasetId } : {}),
        };
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
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: isAppInfoRoute
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
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-blue-600 text-white px-4 py-2 rounded shadow-lg hover:bg-blue-500"
      >
        {isOpen ? "Close Databot" : "💬 Open Databot"}
      </button>

      {isOpen && (
        <div className="mt-2 bg-white border rounded-lg shadow-lg flex flex-col h-[500px]">
          <h2 className="text-lg font-bold p-3 border-b">Databot Tutor</h2>
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
          <form onSubmit={askDatabot} className="p-3 border-t flex">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                location.pathname === "/dashboard" ||
                location.pathname === "/models"
                  ? "Ask about this app…"
                  : botType === "modelbot"
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
