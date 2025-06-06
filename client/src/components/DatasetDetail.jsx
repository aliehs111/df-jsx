// client/src/components/DatasetDetail.jsx
import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/24/outline";

export default function DatasetDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [dataset, setDataset] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const res = await fetch(`/api/datasets/${id}`, {
          method: "GET",
          credentials: "include",
        });
        if (res.status === 401) {
          navigate("/login");
          return;
        }
        if (res.status === 404) {
          navigate("/datasets");
          return;
        }
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Error ${res.status}: ${text}`);
        }
        const data = await res.json();
        setDataset(data);
      } catch (err) {
        console.error("Could not load dataset:", err);
        setError(err.message || "Could not load dataset");
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [id, navigate]);

  const fetchHeatmap = async () => {
    try {
      const res = await fetch(`/api/datasets/${id}/heatmap`, {
        method: "GET",
        credentials: "include",
      });
      if (res.status === 401) {
        navigate("/login");
        return;
      }
      if (!res.ok) {
        return alert("Could not generate heatmap");
      }
      const data = await res.json();
      setHeatmapUrl(data.plot);
    } catch (err) {
      console.error("Heatmap error:", err);
      alert("Could not generate heatmap");
    }
  };

  const fetchInsights = async () => {
    try {
      const res = await fetch(`/api/datasets/${id}/insights`, {
        method: "GET",
        credentials: "include",
      });
      if (res.status === 401) {
        navigate("/login");
        return;
      }
      if (!res.ok) {
        return alert("Could not load insights.");
      }
      const data = await res.json();
      setInsights(data);
    } catch (err) {
      console.error("Insights error:", err);
      alert("Could not load insights.");
    }
  };

  const downloadCleaned = async () => {
    try {
      const res = await fetch(`/api/datasets/${id}/download`, {
        method: "GET",
        credentials: "include",
      });
      if (res.status === 401) {
        navigate("/login");
        return;
      }
      if (!res.ok) {
        return alert("Could not get download link");
      }
      const { url } = await res.json();
      window.open(url, "_blank");
    } catch (err) {
      console.error("Download error:", err);
      alert("Could not get download link");
    }
  };

  if (loading) {
    return <div className="p-6">Loading…</div>;
  }
  if (error) {
    return <div className="p-6 text-red-500">{error}</div>;
  }
  if (!dataset) {
    return <div className="p-6">Dataset not found.</div>;
  }

  const hasClean =
    Array.isArray(dataset.cleaned_data) && dataset.cleaned_data.length > 0;

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-md">
      {/* Title + Processed Badge */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">{dataset.title}</h2>
        {hasClean && (
          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
            Processed
          </span>
        )}
      </div>

      <p className="text-gray-600 mb-2">{dataset.description}</p>
      <p className="text-sm text-gray-400 mb-6">
        Uploaded: {new Date(dataset.uploaded_at).toLocaleString()}
      </p>

      {/* Raw Data Preview */}
      <div className="overflow-auto text-sm bg-gray-100 p-4 rounded mb-6">
        <h3 className="font-semibold mb-2">Raw Data Preview</h3>
        <table className="min-w-full divide-y divide-gray-300 text-xs">
          <thead className="bg-gray-200">
            <tr>
              {Object.keys(dataset.raw_data[0] || {}).map((k) => (
                <th key={k} className="px-2 py-1 text-left font-medium">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-300">
            {dataset.raw_data.slice(0, 5).map((row, rIdx) => (
              <tr key={rIdx}>
                {Object.values(row).map((v, cIdx) => (
                  <td key={cIdx} className="px-2 py-1 whitespace-nowrap">
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Buttons */}
      <div className="mt-6 flex flex-wrap justify-center gap-4">
        <button
          onClick={fetchHeatmap}
          className="bg-blue-800 text-white px-4 py-2 rounded hover:bg-lime-400"
        >
          View Heatmap
        </button>

        <button
          onClick={fetchInsights}
          className="bg-blue-800 text-white px-4 py-2 rounded hover:bg-lime-400"
        >
          View Insights
        </button>
        <Link
          to={`/datasets/${id}/clean`}
          className="bg-blue-800 text-white px-4 py-2 rounded hover:bg-lime-400"
        >
          Pipeline Sandbox
        </Link>

        <Link
          to={`/datasets/${id}/process`}
          className="bg-blue-800 text-white px-4 py-2 rounded hover:bg-indigo-500"
        >
          Preprocess
        </Link>
        <Link
          to="/chat"
          className="
            inline-flex items-center 
            space-x-1 
            bg-lime-500 hover:bg-cyan-700 
            text-white 
            text-xs 
            px-2 py-1 
            rounded
          "
        >
          <ChatBubbleLeftEllipsisIcon className="h-4 w-4" />
          <span>Chat with Databot!</span>
          <img src={newlogo500} alt="Data Tutor" className="h-4 w-4" />
        </Link>
      </div>

      {/* Cleaned Data Preview & Download */}
      {hasClean && (
        <div className="mt-8 bg-gray-50 p-4 rounded-md">
          <h3 className="text-lg font-semibold mb-2">Cleaned Data Preview</h3>
          <div className="overflow-auto mb-4">
            <table className="min-w-full divide-y divide-gray-300 text-xs">
              <thead className="bg-gray-200">
                <tr>
                  {Object.keys(dataset.cleaned_data[0] || {}).map((col) => (
                    <th key={col} className="px-2 py-1 text-left font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-300">
                {dataset.cleaned_data.slice(0, 5).map((row, rIdx) => (
                  <tr key={rIdx}>
                    {Object.values(row).map((v, cIdx) => (
                      <td key={cIdx} className="px-2 py-1 whitespace-nowrap">
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button
            onClick={downloadCleaned}
            className="inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500"
          >
            Download Cleaned CSV
          </button>
        </div>
      )}

      {/* Insights */}
      {insights && (
        <div className="mt-6 space-y-6 bg-gray-50 p-4 rounded-md">
          <h3 className="text-lg font-semibold text-gray-700">
            Dataset Summary
          </h3>
          <p>
            <strong>Shape:</strong> {insights.shape[0]} × {insights.shape[1]}
          </p>
          <p>
            <strong>Columns:</strong> {insights.columns.join(", ")}
          </p>

          <h3 className="font-semibold text-gray-700">Preview Rows</h3>
          <div className="overflow-auto max-h-64 border rounded">
            <table className="min-w-full text-xs">
              <thead className="bg-gray-100 sticky top-0">
                <tr>
                  {Object.keys(insights.preview[0] || {}).map((col) => (
                    <th key={col} className="px-2 py-1 border">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {insights.preview.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((val, j) => (
                      <td key={j} className="px-2 py-1 border">
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 className="font-semibold text-gray-700">Data Types</h3>
          <ul className="list-disc list-inside text-sm">
            {Object.entries(insights.dtypes).map(([col, dt]) => (
              <li key={col}>
                <strong>{col}</strong>: {dt}
              </li>
            ))}
          </ul>

          <h3 className="font-semibold text-gray-700">Missing Values</h3>
          <ul className="list-disc list-inside text-sm">
            {Object.entries(insights.null_counts).map(([col, cnt]) => (
              <li key={col}>
                <strong>{col}</strong>: {cnt}
              </li>
            ))}
          </ul>

          <h3 className="font-semibold text-gray-700">df.info()</h3>
          <pre className="overflow-auto max-h-64 bg-gray-100 p-4 rounded text-xs whitespace-pre-wrap">
            {insights.info_output}
          </pre>
        </div>
      )}

      {/* Heatmap */}
      {heatmapUrl && (
        <img
          src={heatmapUrl}
          alt="Correlation heat-map"
          className="mt-6 rounded shadow-lg"
        />
      )}
    </div>
  );
}
