// client/src/components/DatasetDetail.jsx
import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/24/outline";
import InsightsPanel from "./InsightsPanel";

export default function DatasetDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [dataset, setDataset] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [heatmapError, setHeatmapError] = useState("");
  const [insights, setInsights] = useState(null);
  const [insightsSource, setInsightsSource] = useState("raw");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const hasClean =
    Array.isArray(dataset?.cleaned_data) && dataset.cleaned_data.length > 0;

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const res = await fetch(`/api/datasets/${id}`, {
          credentials: "include",
        });
        if (res.status === 401) return navigate("/login");
        if (res.status === 404) return navigate("/datasets");
        if (!res.ok) throw new Error(`Error ${res.status}`);
        setDataset(await res.json());
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [id, navigate]);

  const fetchHeatmap = async (source = "raw") => {
    setHeatmapError("");
    setHeatmapUrl(null);
    try {
      const res = await fetch(`/api/datasets/${id}/heatmap?which=${source}`, {
        credentials: "include",
      });
      if (res.status === 401) return navigate("/login");
      if (!res.ok) {
        const err = await res.json();
        setHeatmapError(err.error || err.detail);
        return;
      }
      setHeatmapUrl((await res.json()).plot);
    } catch (err) {
      setHeatmapError(err.message);
    }
  };

  const fetchInsights = async (source = "raw") => {
    setInsights(null);
    setInsightsSource(source);
    try {
      const res = await fetch(`/api/datasets/${id}/insights?which=${source}`, {
        credentials: "include",
      });
      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error(`Error ${res.status}`);
      setInsights(await res.json());
    } catch (err) {
      console.error("Insights fetch error:", err);
      alert("Could not load insights.");
    }
  };

  const downloadCleaned = async () => {
    try {
      const res = await fetch(`/api/datasets/${id}/download`, {
        credentials: "include",
      });
      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error("Download failed");
      const { url } = await res.json();
      window.open(url, "_blank");
    } catch (err) {
      console.error("Download error:", err);
      alert("Could not download cleaned data.");
    }
  };
  function renderCell(value) {
    if (typeof value === "boolean") return value ? "True" : "False";
    if (value === null || value === undefined) return "";
    return value;
  }
  if (loading) return <div className="p-6">Loading…</div>;
  if (error) return <div className="p-6 text-red-500">{error}</div>;

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-md">
      {/* Header */}
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

      {/* Data Panels */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
        {/* Raw Data Panel */}
        <div className="p-4 border rounded">
          <h4 className="font-semibold mb-2">Raw Data Preview</h4>
          <div className="overflow-auto text-xs bg-gray-100 p-2 rounded mb-4 max-h-40">
            <table className="min-w-full">
              <thead>
                <tr>
                  {Object.keys(dataset.raw_data[0] || {}).map((col) => (
                    <th key={col} className="px-1 py-0.5 font-medium text-left">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataset.raw_data.slice(0, 3).map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="px-1 py-0.5 whitespace-nowrap">
                        {renderCell(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => fetchHeatmap("raw")}
              className="bg-blue-800 text-white px-3 py-1 rounded"
            >
              Heatmap
            </button>
            <button
              onClick={() => fetchInsights("raw")}
              className="bg-blue-800 text-white px-3 py-1 rounded"
            >
              Insights
            </button>
            <Link
              to={`/datasets/${id}/clean?which=raw`}
              className="bg-blue-600 text-white px-3 py-1 rounded"
            >
              Sandbox
            </Link>
          </div>
        </div>

        {/* Cleaned Data Panel */}
        {hasClean && (
          <div className="p-4 border rounded">
            <h4 className="font-semibold mb-2">Cleaned Data Preview</h4>
            <div className="overflow-auto text-xs	bg-gray-100	p-2 rounded mb-4 max-h-40">
              <table className="min-w-full">
                <thead>
                  <tr>
                    {Object.keys(dataset.cleaned_data[0] || {}).map((col) => (
                      <th
                        key={col}
                        className="px-1 py-0.5 font-medium text-left"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.cleaned_data.slice(0, 3).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-1 py-0.5 whitespace-nowrap">
                          {renderCell(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => fetchHeatmap("cleaned")}
                className="bg-green-800 text-white px-3 py-1 rounded"
              >
                Heatmap
              </button>
              <button
                onClick={() => fetchInsights("cleaned")}
                className="bg-green-800 text-white px-3 py-1 rounded"
              >
                Insights
              </button>
              <Link
                to={`/datasets/${id}/clean?which=cleaned`}
                className="bg-green-600 text-white px-3 py-1 rounded"
              >
                Sandbox
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* Insights Section */}
      {insights && (
        <>
          <div className="mb-2 text-sm text-gray-600">
            Showing insights for{" "}
            <strong>
              {insightsSource === "raw" ? "Raw Data" : "Cleaned Data"}
            </strong>
          </div>
          <InsightsPanel insights={insights} />
        </>
      )}

      {/* Heatmap Section */}
      {heatmapError && (
        <div className="mt-4 p-3 bg-red-100 text-red-800 rounded">
          {heatmapError}
        </div>
      )}
      {heatmapUrl && (
        <img src={heatmapUrl} alt="heatmap" className="mt-4 rounded shadow" />
      )}

      {/* Chat Link */}
      <div className="mt-8 text-center">
        <Link
          to="/chat"
          className="inline-flex items-center space-x-1 bg-lime-500 hover:bg-cyan-700 text-white text-xs px-2 py-1 rounded"
        >
          <ChatBubbleLeftEllipsisIcon className="h-4 w-4" />
          <span>Chat with Databot!</span>
          <img src={newlogo500} alt="Data Tutor" className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}
