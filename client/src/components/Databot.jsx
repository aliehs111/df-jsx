import { useState } from "react";

export default function Databot({ selectedDataset }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const askDatabot = async () => {
    if (!question) return;
    setIsLoading(true);
    try {
      const res = await fetch("/api/databot/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ dataset_id: selectedDataset, question }),
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
            disabled={isLoading || !selectedDataset}
          >
            {isLoading ? "Thinking..." : "Ask"}
          </button>
          {answer && (
            <div className="mt-3 p-2 bg-gray-100 border rounded text-sm">
              <strong>Databot:</strong> {answer}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
