// client/src/components/DatasetDetail.jsx
import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/24/outline";
import InsightsPanel from "./InsightsPanel";
import MetalButton from "./MetalButton";
import { ArrowDownTrayIcon } from "@heroicons/react/24/outline";

export default function DatasetDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [dataset, setDataset] = useState(null);
  const [cleanedPreview, setCleanedPreview] = useState(null);
  const [cleanedPreviewError, setCleanedPreviewError] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [heatmapError, setHeatmapError] = useState("");
  const [insights, setInsights] = useState(null);
  const [insightsSource, setInsightsSource] = useState("raw");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const hasClean = dataset?.has_cleaned_data || false;

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const res = await fetch(`/api/datasets/${id}`, {
          credentials: "include",
        });
        if (res.status === 401) return navigate("/login");
        if (res.status === 404) return navigate("/datasets");
        if (!res.ok) throw new Error(`Error ${res.status}`);
        const data = await res.json();
        setDataset(data);

        if (data.s3_key_cleaned && data.has_cleaned_data) {
          try {
            const cleanRes = await fetch(
              `/api/datasets/${id}/insights?which=cleaned`,
              {
                credentials: "include",
              }
            );
            if (!cleanRes.ok) {
              const err = await cleanRes
                .json()
                .catch(() => ({ detail: "Failed to load cleaned data" }));
              throw new Error(err.detail || `Error ${cleanRes.status}`);
            }
            const cleanData = await cleanRes.json();
            setCleanedPreview(cleanData.preview);
          } catch (err) {
            setCleanedPreviewError(err.message);
            console.error("Failed to load cleaned preview:", err);
          }
        }
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
      <header className="relative overflow-hidden bg-gradient-to-r from-primary via-primary/90 to-secondary py-8 px-6 sm:px-12 shadow-md mb-8 rounded-xl">
        {/* subtle glow */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-10 -right-10 h-44 w-44 rounded-full bg-white/10 blur-2xl"
        />
        <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          {/* Left: title + meta pills */}
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white drop-shadow-sm">
                {dataset?.title ?? "Dataset"}
              </h1>
              {dataset?.id != null && (
                <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                  ID #{dataset.id}
                </span>
              )}
              {typeof dataset?.n_rows === "number" &&
                typeof dataset?.n_columns === "number" && (
                  <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                    {dataset.n_rows.toLocaleString()} rows × {dataset.n_columns}{" "}
                    cols
                  </span>
                )}
              {typeof dataset?.has_missing_values === "boolean" && (
                <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                  {dataset.has_missing_values ? "Missing values" : "No missing"}
                </span>
              )}
            </div>
            {dataset?.description && (
              <p className="mt-1 text-cyan-100/90 text-sm line-clamp-2">
                {dataset.description}
              </p>
            )}
          </div>

          {/* Right: actions */}
          <div className="flex items-center gap-2">
            {hasClean ? (
              <MetalButton
                tone="steel"
                onClick={downloadCleaned}
                className="rounded-full gap-2"
              >
                <ArrowDownTrayIcon className="h-4 w-4" />
                Download Cleaned Dataset
              </MetalButton>
            ) : (
              <span className="rounded-full bg-white/10 px-3 py-1 text-sm text-white/90 ring-1 ring-white/20">
                Not cleaned yet
              </span>
            )}
          </div>
        </div>
      </header>

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
                  {dataset.preview_data && dataset.preview_data.length > 0 ? (
                    Object.keys(dataset.preview_data[0] || {}).map((col) => (
                      <th
                        key={col}
                        className="px-1 py-0.5 font-medium text-left"
                      >
                        {col}
                      </th>
                    ))
                  ) : (
                    <th>No data available</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {dataset.preview_data && dataset.preview_data.length > 0 ? (
                  dataset.preview_data.slice(0, 3).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-1 py-0.5 whitespace-nowrap">
                          {renderCell(v)}
                        </td>
                      ))}
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td>No data available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap gap-2">
            <MetalButton tone="gold" onClick={() => fetchHeatmap("raw")}>
              Heatmap
            </MetalButton>

            <MetalButton tone="blueSteel" onClick={() => fetchInsights("raw")}>
              Insights
            </MetalButton>

            <MetalButton tone="steel" to={`/datasets/${id}/clean?which=raw`}>
              Preprocess
            </MetalButton>
          </div>
        </div>

        {/* Cleaned Data Panel */}
        {hasClean && (
          <div className="p-4 border rounded">
            <h4 className="font-semibold mb-2">Cleaned Data Preview</h4>
            <div className="overflow-auto text-xs bg-gray-100 p-2 rounded mb-4 max-h-40">
              {cleanedPreviewError ? (
                <div className="text-red-600">
                  Error loading cleaned data: {cleanedPreviewError}
                </div>
              ) : (
                <table className="min-w-full">
                  <thead>
                    <tr>
                      {cleanedPreview && cleanedPreview.length > 0 ? (
                        Object.keys(cleanedPreview[0] || {}).map((col) => (
                          <th
                            key={col}
                            className="px-1 py-0.5 font-medium text-left"
                          >
                            {col}
                          </th>
                        ))
                      ) : (
                        <th>No cleaned data available</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {cleanedPreview && cleanedPreview.length > 0 ? (
                      cleanedPreview.slice(0, 3).map((row, i) => (
                        <tr key={i}>
                          {Object.values(row).map((v, j) => (
                            <td
                              key={j}
                              className="px-1 py-0.5 whitespace-nowrap"
                            >
                              {renderCell(v)}
                            </td>
                          ))}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td>No cleaned data available</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <MetalButton tone="gold" onClick={() => fetchHeatmap("cleaned")}>
                Heatmap
              </MetalButton>

              <MetalButton
                tone="blueSteel"
                onClick={() => fetchInsights("cleaned")}
              >
                Insights
              </MetalButton>

              <MetalButton
                tone="titanium"
                to={`/datasets/${id}/clean?which=cleaned`}
              >
                Process Again
              </MetalButton>
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
    </div>
  );
}
