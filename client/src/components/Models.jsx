import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import DevNotesModels from "../components/DevNotesModels.jsx";

export default function Models() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [result, setResult] = useState(null);
  const [columns, setColumns] = useState([]);
  const [stringColumns, setStringColumns] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState("");
  const [C, setC] = useState(1.0);
  const [targetUniqueCount, setTargetUniqueCount] = useState(null);
  const [models, setModels] = useState([]);
  const [nClusters, setNClusters] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // Hide unfinished models from the UI
  const EXCLUDED_MODELS = new Set(["TimeSeriesForecasting"]);
  const visibleModels = models.filter((m) => !EXCLUDED_MODELS.has(m.name));

  useEffect(() => {
    if (visibleModels.length > 0 && !selectedModel) {
      setSelectedModel(visibleModels[0].name);
    } else if (selectedModel && EXCLUDED_MODELS.has(selectedModel)) {
      // If an excluded model was previously selected, switch to the first visible one
      setSelectedModel(visibleModels[0]?.name || null);
    }
  }, [models, visibleModels, selectedModel]);

  const modelMeta = {
    RandomForest: {
      tags: ["Target required", "≥2 classes", "Numeric+Categorical OK"],
      hint: "Supervised. Choose a target with at least 2 classes. Handles mixed features; clean missing values first.",
    },
    LogisticRegression: {
      tags: ["Binary target", "Target required", "Numeric+Categorical OK"],
      hint: "Supervised (best for binary). Needs a clean target column (0/1 or two classes).",
    },
    PCA_KMeans: {
      tags: ["No target", "Numeric features", "Choose K"],
      hint: "Unsupervised. No target column. Provide numeric features; standardization recommended.",
    },
    Sentiment: {
      tags: ["GPU Inference", "Text column", "English"],
      hint: "Needs a text column. Returns label counts and sample scores.",
    },
    TimeSeriesForecasting: {
      tags: ["Date+Value columns", "Regular frequency"],
      hint: "Provide date and value columns. Ensure clean, sorted, regular intervals.",
    },
    AnomalyDetection: {
      tags: ["GPU Inference", "No target", "Numeric features"],
      hint: "Unsupervised. Provide numeric features; works best after cleaning and scaling.",
    },
  };

  useEffect(() => {
    const fetchCleanedDatasets = async () => {
      try {
        const res = await fetch("/api/datasets/cleaned", {
          method: "GET",
          credentials: "include",
        });
        if (res.status === 401) {
          navigate("/login");
          return;
        }
        if (!res.ok) {
          console.error("Failed to fetch cleaned datasets:", res.status);
          setDatasets([]);
          return;
        }
        const data = await res.json();
        setDatasets(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Failed to fetch cleaned datasets", err);
        setDatasets([]);
      }
    };
    fetchCleanedDatasets();
  }, [navigate]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch("/api/models/available", {
          method: "GET",
          credentials: "include",
        });
        if (!res.ok) {
          console.error("Failed to fetch models:", res.status);
          setModels([]);
          return;
        }
        const data = await res.json();
        setModels(data.models || []);
      } catch (err) {
        console.error("Failed to fetch models", err);
        setModels([]);
      }
    };
    fetchModels();
  }, []);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].name);
    }
  }, [models, selectedModel]);

  useEffect(() => {
    const fetchColumns = async () => {
      if (!selectedDataset) {
        setColumns([]);
        setStringColumns([]);
        setSelectedTarget("");
        setTargetUniqueCount(null);
        return;
      }
      try {
        const res = await fetch(`/api/datasets/${selectedDataset}/columns`, {
          method: "GET",
          credentials: "include",
        });
        if (res.status === 401) {
          navigate("/login");
          return;
        }
        if (!res.ok) {
          const data = await res.json();
          console.error("Failed to fetch columns:", res.status, data);
          setColumns([]);
          setStringColumns([]);
          setSelectedTarget("");
          setTargetUniqueCount(null);
          setResult({
            error:
              data.detail ||
              "Failed to load dataset columns. Please ensure the dataset is properly cleaned.",
          });
          return;
        }
        const data = await res.json();
        setColumns(Array.isArray(data.columns) ? data.columns : []);
        setStringColumns(data.columns); // Update when backend provides dtypes
        setSelectedTarget("");
        setTargetUniqueCount(null);
        setResult(null);
      } catch (err) {
        console.error("Failed to fetch columns", err);
        setColumns([]);
        setStringColumns([]);
        setSelectedTarget("");
        setTargetUniqueCount(null);
        setResult({
          error: `Failed to load columns: ${err.message}. Please check Data Cleaning.`,
        });
      }
    };
    fetchColumns();
  }, [selectedDataset, navigate]);

  useEffect(() => {
    const fetchUniqueCount = async () => {
      if (
        !selectedDataset ||
        !selectedTarget ||
        selectedModel === "Sentiment"
      ) {
        setTargetUniqueCount(null);
        return;
      }
      try {
        const res = await fetch(
          `/api/datasets/${selectedDataset}/column/${encodeURIComponent(
            selectedTarget
          )}/unique`,
          { method: "GET", credentials: "include" }
        );
        if (res.status === 401) {
          navigate("/login");
          return;
        }
        if (!res.ok) {
          console.error("Failed to fetch unique count:", res.status);
          setTargetUniqueCount(null);
          return;
        }
        const data = await res.json();
        setTargetUniqueCount(data.unique_count);
      } catch (err) {
        console.error("Failed to fetch unique count", err);
        setTargetUniqueCount(null);
      }
    };
    fetchUniqueCount();
  }, [selectedDataset, selectedTarget, selectedModel, navigate]);

  const isTargetValid =
    selectedModel === "PCA_KMeans" ||
    selectedModel === "Sentiment" ||
    selectedModel === "AnomalyDetection" ||
    selectedModel === "TimeSeriesForecasting" ||
    (selectedTarget && targetUniqueCount >= 2);

  const handleRunModel = async () => {
    if (!selectedDataset || !selectedModel) return;

    if (
      (selectedModel === "RandomForest" ||
        selectedModel === "LogisticRegression" ||
        selectedModel === "Sentiment") &&
      !selectedTarget
    ) {
      setResult({
        error: `Please select a ${
          selectedModel === "Sentiment" ? "text" : "target"
        } column.`,
      });
      return;
    }

    if (
      (selectedModel === "RandomForest" ||
        selectedModel === "LogisticRegression") &&
      targetUniqueCount < 2
    ) {
      setResult({
        error: `Target column '${selectedTarget}' has only ${targetUniqueCount} class(es). At least 2 required.`,
      });
      return;
    }

    if (
      selectedModel === "TimeSeriesForecasting" &&
      (!selectedTarget || !selectedTarget.includes("|"))
    ) {
      setResult({
        error: "Please select both a date and value column for forecasting.",
      });
      return;
    }

    setIsLoading(true);
    try {
      const payload = {
        dataset_id: selectedDataset,
        model_name: selectedModel,
      };
      if (selectedModel === "RandomForest") {
        payload.target_column = selectedTarget;
        payload.n_estimators = nEstimators;
        if (maxDepth) payload.max_depth = parseInt(maxDepth);
      } else if (selectedModel === "LogisticRegression") {
        payload.target_column = selectedTarget;
        payload.C = C;
      } else if (selectedModel === "PCA_KMeans") {
        payload.n_clusters = nClusters;
      } else if (selectedModel === "Sentiment") {
        payload.target_column = selectedTarget;
      } else if (selectedModel === "AnomalyDetection") {
        // Backend handles records
      } else if (selectedModel === "TimeSeriesForecasting") {
        payload.target_column = selectedTarget; // "date|value"
      }

      const res = await fetch("/api/models/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        console.error("Non-JSON response:", text.slice(0, 200));
        alert("Backend error: Non-JSON response received.");
        return;
      }

      if (!res.ok) {
        setResult({ error: data.detail || `Model run failed (${res.status})` });
      } else {
        setResult(data);
      }
    } catch (err) {
      setResult({ error: `Network error: ${err.message}` });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-8 bg-gray-50 min-h-screen">
      {/* Header */}
      <header className="relative overflow-hidden bg-gradient-to-r from-primary via-primary/90 to-secondary py-10 px-8 sm:px-20 shadow-md mb-8">
        <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white drop-shadow-sm">
                Models
              </h1>
              <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                v0.9 • dev
              </span>
            </div>
            <p className="mt-2 text-cyan-100 text-sm">
              Random Forest, Logistic Regression, PCA + KMeans, Sentiment
              Analysis, Anomaly Detection
            </p>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4 flex items-center">
            <svg
              className="w-6 h-6 mr-3 text-blue-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
              />
            </svg>
            Select Cleaned Dataset
          </h2>
          {datasets.length === 0 ? (
            <p className="text-gray-500 italic text-lg">
              No cleaned datasets available. Please upload and clean a dataset
              first.
            </p>
          ) : (
            <ul className="space-y-3">
              {datasets.map((ds) => (
                <li
                  key={ds.id}
                  className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                    selectedDataset === ds.id
                      ? "bg-blue-100 border-blue-400 shadow-inner"
                      : "border-gray-200 hover:bg-blue-50 hover:shadow-md"
                  } border`}
                  onClick={() => setSelectedDataset(ds.id)}
                >
                  <span className="text-gray-800 font-medium text-lg">
                    {ds.title}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4 flex items-center">
            <svg
              className="w-6 h-6 mr-3 text-green-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 6V4m0 2v.01M12 18v2m0-2v-.01M6 12H4m2 0h.01M20 12h-2m2 0h-.01M9 3h6v18H9V3z"
              />
            </svg>
            Configure Model
          </h2>
          <ul className="space-y-3 mb-6">
            {visibleModels.map((model) => (
              <li
                key={model.name}
                className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                  selectedModel === model.name
                    ? "bg-green-100 border-green-400 shadow-inner"
                    : "border-gray-200 hover:bg-green-50 hover:shadow-md"
                } border`}
                onClick={() => setSelectedModel(model.name)}
                title={model.description || ""}
              >
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-gray-800 font-medium text-lg">
                      {model.name}
                    </span>
                    {modelMeta[model.name]?.tags?.map((tag, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center rounded-full bg-gray-100 text-gray-700 px-2 py-0.5 text-[10px] font-semibold ring-1 ring-gray-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  {modelMeta[model.name]?.hint && (
                    <div className="mt-1 text-xs text-gray-600 leading-snug">
                      {modelMeta[model.name].hint}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>

          {(selectedModel === "RandomForest" ||
            selectedModel === "LogisticRegression" ||
            selectedModel === "Sentiment") &&
            columns.length > 0 && (
              <div className="mb-6">
                <label
                  htmlFor="target"
                  className="block text-base font-medium text-gray-700 mb-2"
                >
                  {selectedModel === "Sentiment"
                    ? "Text Column"
                    : "Target Column"}
                </label>
                <select
                  id="target"
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
                >
                  <option value="">
                    Select a {selectedModel === "Sentiment" ? "text" : "target"}{" "}
                    column
                  </option>
                  {(selectedModel === "Sentiment"
                    ? stringColumns
                    : columns
                  ).map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                {selectedTarget &&
                  targetUniqueCount !== null &&
                  targetUniqueCount < 2 &&
                  selectedModel !== "Sentiment" && (
                    <p className="mt-3 text-sm text-red-600 font-medium">
                      ⚠️ Target column has only {targetUniqueCount} class(es).
                      At least 2 required.
                    </p>
                  )}
              </div>
            )}

          {selectedModel === "RandomForest" && (
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <label
                  htmlFor="n_estimators"
                  className="block text-base font-medium text-gray-700 mb-2"
                >
                  Number of Trees
                </label>
                <input
                  id="n_estimators"
                  type="number"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(parseInt(e.target.value))}
                  min="10"
                  step="10"
                  className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
                />
              </div>
              <div>
                <label
                  htmlFor="max_depth"
                  className="block text-base font-medium text-gray-700 mb-2"
                >
                  Max Depth (Optional)
                </label>
                <input
                  id="max_depth"
                  type="number"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(e.target.value)}
                  min="1"
                  placeholder="None"
                  className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
                />
              </div>
            </div>
          )}

          {selectedModel === "LogisticRegression" && (
            <div className="mb-6">
              <label
                htmlFor="C"
                className="block text-base font-medium text-gray-700 mb-2"
              >
                Regularization Strength (C)
              </label>
              <input
                id="C"
                type="number"
                value={C}
                onChange={(e) => setC(parseFloat(e.target.value))}
                min="0.01"
                step="0.01"
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
              />
            </div>
          )}

          {selectedModel === "PCA_KMeans" && (
            <div className="mb-6">
              <label
                htmlFor="clusters"
                className="block text-base font-medium text-gray-700 mb-2"
              >
                Number of Clusters
              </label>
              <select
                id="clusters"
                value={nClusters}
                onChange={(e) => setNClusters(parseInt(e.target.value))}
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
              >
                {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((val) => (
                  <option key={val} value={val}>
                    {val}
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedModel === "TimeSeriesForecasting" && columns.length > 0 && (
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div>
                <label
                  htmlFor="dateColumn"
                  className="block text-base font-medium text-gray-700 mb-2"
                >
                  Date Column
                </label>
                <select
                  id="dateColumn"
                  value={selectedTarget.split("|")[0] || ""}
                  onChange={(e) => {
                    const valueCol = selectedTarget.split("|")[1] || "";
                    setSelectedTarget(`${e.target.value}|${valueCol}`);
                  }}
                  className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
                >
                  <option value="">Select a date column</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label
                  htmlFor="valueColumn"
                  className="block text-base font-medium text-gray-700 mb-2"
                >
                  Value Column
                </label>
                <select
                  id="valueColumn"
                  value={selectedTarget.split("|")[1] || ""}
                  onChange={(e) => {
                    const dateCol = selectedTarget.split("|")[0] || "";
                    setSelectedTarget(`${dateCol}|${e.target.value}`);
                  }}
                  className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500 text-base py-2 px-3"
                >
                  <option value="">Select a value column</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              {selectedTarget && !selectedTarget.includes("|") && (
                <p className="col-span-2 mt-3 text-sm text-red-600 font-medium">
                  ⚠️ Please select both a date and value column.
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="mt-10 text-center">
        <button
          onClick={handleRunModel}
          className={`px-8 py-4 rounded-lg text-white font-semibold text-lg transition-all duration-300 ${
            isLoading || !selectedDataset || !selectedModel || !isTargetValid
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 hover:shadow-lg"
          }`}
          disabled={
            isLoading || !selectedDataset || !selectedModel || !isTargetValid
          }
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin h-6 w-6 mr-3 text-white"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Running Model...
            </span>
          ) : (
            "Execute Model"
          )}
        </button>
      </div>

      {result && (
        <div className="mt-10 bg-white p-8 rounded-xl shadow-md border border-gray-200">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 flex items-center">
            <svg
              className="w-7 h-7 mr-3 text-indigo-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            Model Execution Results
          </h2>

          {result.error && (
            <div className="mb-8 p-6 bg-red-50 border border-red-300 rounded-lg shadow-inner">
              <p className="text-red-700 font-semibold text-lg flex items-center">
                <svg
                  className="w-6 h-6 mr-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Error: {result.error}
              </p>
              {(result.error.includes("preprocess the dataset") ||
                result.error.includes("NoSuchKey") ||
                result.error.includes("Missing target_column") ||
                result.error.includes("Missing date_column|value_column") ||
                result.error.includes("No valid numeric feature columns") ||
                result.error.includes("Empty dataset") ||
                result.error.includes("No text data")) && (
                <p className="mt-3 text-base text-red-600">
                  {result.error.includes("NoSuchKey")
                    ? "The cleaned dataset file is missing. Re-clean the dataset or upload a new file."
                    : result.error.includes("preprocess the dataset")
                    ? "The dataset has missing values. Use the Data Cleaning page to address this."
                    : result.error.includes("Missing target_column")
                    ? "Please select a valid target column."
                    : result.error.includes("Missing date_column|value_column")
                    ? "Please select both a date and value column."
                    : result.error.includes("Empty dataset")
                    ? "The dataset is empty. Upload a valid dataset."
                    : result.error.includes("No text data")
                    ? "No valid text in this column. Select a different column."
                    : "Insufficient numeric features. Clean or transform the data."}
                  <br />
                  Consult the chatbot for data preparation guidance.
                </p>
              )}
            </div>
          )}

          {result.image_base64 && (
            <div className="mb-8 text-center">
              <h3 className="text-xl font-semibold text-gray-700 mb-4">
                Visualization
              </h3>
              <img
                src={`data:image/png;base64,${result.image_base64}`}
                alt="Model visualization"
                className="max-w-full h-auto rounded-lg shadow-lg mx-auto"
              />
            </div>
          )}

          {!result.error &&
            !result.image_base64 &&
            result.model !== "AnomalyDetection" && (
              <p className="mb-8 text-gray-500 text-lg italic text-center">
                No visualization generated for this model run. Review the
                detailed metrics below.
              </p>
            )}

          {result.message && (
            <p className="mb-8 text-green-600 text-lg font-medium flex items-center">
              <svg
                className="w-6 h-6 mr-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M5 13l4 4L19 7"
                />
              </svg>
              {result.message}
            </p>
          )}

          {result.model === "PCA_KMeans" && (
            <div className="space-y-8">
              {result.n_clusters && (
                <p className="text-gray-700 text-lg">
                  <span className="font-semibold">Number of Clusters:</span>{" "}
                  {result.n_clusters}
                </p>
              )}
              {result.cluster_counts && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Cluster Distribution
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Cluster
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Count
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(result.cluster_counts).map(
                          ([cluster, count]) => (
                            <tr key={cluster}>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {cluster}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {count}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {result.pca_variance_ratio && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    PCA Explained Variance
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Component
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Variance (%)
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {result.pca_variance_ratio.map((val, i) => (
                          <tr key={i}>
                            <td className="px-6 py-3 text-sm text-gray-600">
                              PC{i + 1}
                            </td>
                            <td className="px-6 py-3 text-sm text-gray-600">
                              {(val * 100).toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {(result.model === "RandomForest" ||
            result.model === "LogisticRegression") && (
            <div className="space-y-8">
              {result.class_counts && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Class Distribution
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Class
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Count
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(result.class_counts).map(
                          ([cls, count]) => (
                            <tr key={cls}>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {cls}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {count}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {result.classification_report && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Classification Metrics
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Class
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Precision
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Recall
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            F1-Score
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Support
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(result.classification_report)
                          .filter(
                            ([key]) =>
                              ![
                                "accuracy",
                                "macro avg",
                                "weighted avg",
                              ].includes(key)
                          )
                          .map(([cls, metrics]) => (
                            <tr key={cls}>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {cls}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {metrics.precision.toFixed(2)}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {metrics.recall.toFixed(2)}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {metrics["f1-score"].toFixed(2)}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {metrics.support}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                      <tfoot className="bg-gray-50 divide-y divide-gray-200">
                        {["macro avg", "weighted avg"].map(
                          (avgType) =>
                            result.classification_report[avgType] && (
                              <tr key={avgType}>
                                <td className="px-6 py-3 text-sm font-medium text-gray-700">
                                  {avgType}
                                </td>
                                <td className="px-6 py-3 text-sm text-gray-600">
                                  {result.classification_report[
                                    avgType
                                  ].precision.toFixed(2)}
                                </td>
                                <td className="px-6 py-3 text-sm text-gray-600">
                                  {result.classification_report[
                                    avgType
                                  ].recall.toFixed(2)}
                                </td>
                                <td className="px-6 py-3 text-sm text-gray-600">
                                  {result.classification_report[avgType][
                                    "f1-score"
                                  ].toFixed(2)}
                                </td>
                                <td className="px-6 py-3 text-sm text-gray-600">
                                  {
                                    result.classification_report[avgType]
                                      .support
                                  }
                                </td>
                              </tr>
                            )
                        )}
                        {result.classification_report.accuracy !==
                          undefined && (
                          <tr>
                            <td className="px-6 py-3 text-sm font-medium text-gray-700">
                              Accuracy
                            </td>
                            <td
                              colSpan="4"
                              className="px-6 py-3 text-sm text-gray-600 text-center"
                            >
                              {result.classification_report.accuracy.toFixed(2)}
                            </td>
                          </tr>
                        )}
                      </tfoot>
                    </table>
                  </div>
                </div>
              )}
              {result.confusion_matrix && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Confusion Matrix
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Actual \ Predicted
                          </th>
                          {result.confusion_matrix[0].map((_, i) => (
                            <th
                              key={i}
                              className="px-6 py-3 text-left text-sm font-semibold text-gray-700"
                            >
                              {i}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {result.confusion_matrix.map((row, i) => (
                          <tr key={i}>
                            <td className="px-6 py-3 text-sm font-medium text-gray-700">
                              {i}
                            </td>
                            {row.map((val, j) => (
                              <td
                                key={j}
                                className="px-6 py-3 text-sm text-gray-600"
                              >
                                {val}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {result.feature_importances &&
                result.model === "RandomForest" && (
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">
                      Feature Importances
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                              Feature
                            </th>
                            <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                              Importance
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                          {result.feature_importances.map((val, i) => (
                            <tr key={i}>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                Feature {i + 1}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {val.toFixed(4)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              {result.coefficients && result.model === "LogisticRegression" && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Coefficients
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Class
                          </th>
                          {result.coefficients[0].map((_, j) => (
                            <th
                              key={j}
                              className="px-6 py-3 text-left text-sm font-semibold text-gray-700"
                            >
                              Feature {j + 1}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {result.coefficients.map((row, i) => (
                          <tr key={i}>
                            <td className="px-6 py-3 text-sm text-gray-600">
                              {i}
                            </td>
                            {row.map((val, j) => (
                              <td
                                key={j}
                                className="px-6 py-3 text-sm text-gray-600"
                              >
                                {val.toFixed(4)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {result.model === "Sentiment" && (
            <div className="space-y-8">
              {result.results && (
                <p className="text-gray-700 text-lg">
                  <span className="font-semibold">Processed Texts:</span>{" "}
                  {result.results.length}
                </p>
              )}
              {result.sentiment_counts && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Sentiment Counts
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Sentiment
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Count
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {Object.entries(result.sentiment_counts).map(
                          ([label, count]) => (
                            <tr key={label}>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {label}
                              </td>
                              <td className="px-6 py-3 text-sm text-gray-600">
                                {count}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {result.results && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    Sample Results
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Sentiment
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">
                            Score
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {result.results.slice(0, 10).map((r, idx) => (
                          <tr key={idx}>
                            <td className="px-6 py-3 text-sm text-gray-600">
                              {r.label}
                            </td>
                            <td className="px-6 py-3 text-sm text-gray-600">
                              {r.score.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-3 text-sm text-gray-500">
                    Showing first 10 results. Original texts not included in
                    response for brevity.
                  </p>
                </div>
              )}
            </div>
          )}

          {result.model === "AnomalyDetection" && (
            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Detected Anomalies
              </h3>
              {result.anomalies?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white border border-gray-200 rounded-lg shadow-sm divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {Object.keys(result.anomalies[0]).map((key) => (
                          <th
                            key={key}
                            className="px-6 py-3 text-left text-sm font-semibold text-gray-700"
                          >
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {result.anomalies.map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          {Object.values(row).map((val, j) => (
                            <td
                              key={j}
                              className="px-6 py-3 text-sm text-gray-600"
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {result.n_anomalies > result.anomalies.length && (
                    <p className="mt-3 text-sm text-gray-500">
                      Showing first {result.anomalies.length} of{" "}
                      {result.n_anomalies} anomalies.
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-gray-500 text-lg italic">
                  No anomalies detected in the dataset.
                </p>
              )}
              {result.n_records && (
                <p className="mt-6 text-gray-700 text-lg">
                  <span className="font-semibold">
                    Total Records Processed:
                  </span>{" "}
                  {result.n_records}
                </p>
              )}
            </div>
          )}
        </div>
      )}
      <DevNotesModels />
    </div>
  );
}
