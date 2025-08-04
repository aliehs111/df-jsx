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

  const askDatabot = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/databot/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ dataset_id: datasetId, question: input }),
      });

      const data = await res.json();
      const assistantMessage = res.ok
        ? { role: "assistant", content: data.answer }
        : {
            role: "assistant",
            content: "Error: " + (data.detail || "Unknown error"),
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
              placeholder="Ask about your dataset..."
              className="flex-grow border rounded-l px-3 py-2"
              disabled={!datasetId}
            />
            <button
              type="submit"
              className="bg-green-600 text-white px-4 rounded-r hover:bg-green-500"
              disabled={isLoading || !datasetId}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
