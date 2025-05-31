// client/src/components/DataInsights.jsx
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

export default function DataInsights() {
 const { id } = useParams();
const datasetId = id || 1; 
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);


  useEffect(() => {
    // Simulated insight text while backend is unavailable
    setTimeout(() => {
      setInsights(`📊 AI-Generated Insights for Dataset ${id}

- 'Age' has 12% missing values.
- 'Income' is skewed with outliers over $1M.
- 'Has_Debt' is correlated with 'Target' (r = 0.52).
- Suggested: impute 'Age', log-transform 'Income', encode 'Has_Debt'.`);
      setLoading(false);
    }, 500);
  }, [id]);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-cyan-200 py-6 px-20">
        <h1 className="text-4xl font-bold text-white">AI Insights</h1>
        <p className="text-blue-100 mt-1 text-lg">For dataset ID: {id}</p>
      </header>

      <main className="-mt-6 mx-auto max-w-4xl px-6 py-10 space-y-6 bg-cyan-50">
        <div className="bg-white rounded-lg shadow p-6">
          {loading ? (
            <p className="text-gray-600">⏳ Generating insights...</p>
          ) : error ? (
            <p className="text-red-600">{error}</p>
          ) : (
            <pre className="whitespace-pre-wrap text-sm text-gray-800 leading-relaxed">
              {insights}
            </pre>
          )}
        </div>
      </main>
    </div>
  );
}
