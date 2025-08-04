import { useState } from "react";
import { useLocation } from "react-router-dom";

export default function Databot({ selectedDataset }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // 👇 Detect datasetId from the URL (e.g., /datasets/22)
  const location = useLocation();
  const match = location.pathname.match(/\/datasets\/(\d+)/);
  const datasetId = match ? parseInt(match[1]) : selectedDataset || null;

  console.log("Databot datasetId:", datasetId);

  const API_BASE =
    import.meta.env.MODE === "development" ? "http://127.0.0.1:8000" : "";

  const askDatabot = async () => {
    if (!question) return;
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/databot/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ dataset_id: datasetId, question }),
      });

      const data = await res.json();
      if (!res.ok) {
        setAnswer("Error: " + (data.detail || "Unknown error"));
      } else {
        setAnswer(data.answer);
      }
    } catch (err) {
      setAnswer("Error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-4 right-4 w-80 z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-blue-600 text-white px-4 py-2 rounded shadow-lg hover:bg-blue-500"
      >
        {isOpen ? "Close Databot" : "💬 Open Databot"}
      </button>

      {isOpen && (
        <div className="mt-2 bg-white border rounded-lg shadow-lg p-4">
          <h2 className="text-lg font-bold mb-2">Databot Tutor</h2>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about your dataset..."
            className="w-full p-2 border rounded mb-2"
            rows={3}
          />
          <button
            onClick={askDatabot}
            className="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-500"
            disabled={isLoading || !datasetId}
          >
            {isLoading ? "Thinking..." : "Ask"}
          </button>
          {answer && (
            <div className="mt-3 p-2 bg-gray-100 border rounded text-sm max-h-48 overflow-y-auto">
              <strong>Databot:</strong> {answer}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
