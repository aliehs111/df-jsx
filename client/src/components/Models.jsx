import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Models() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForest");
  const [result, setResult] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState("");
  const [C, setC] = useState(1.0);
  const [targetUniqueCount, setTargetUniqueCount] = useState(null);
  const models = ["RandomForest", "PCA_KMeans", "LogisticRegression"];
  const [nClusters, setNClusters] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

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
        console.log("Debug: Fetched cleaned datasets:", data);
        setDatasets(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Failed to fetch cleaned datasets", err);
        setDatasets([]);
      }
    };
    fetchCleanedDatasets();
  }, [navigate]);

  useEffect(() => {
    const fetchColumns = async () => {
      if (!selectedDataset) {
        setColumns([]);
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
          setSelectedTarget("");
          setTargetUniqueCount(null);
          setResult({
            error:
              data.detail ||
              "Failed to load dataset columns. Please ensure the dataset is properly cleaned in Data Cleaning.",
          });
          return;
        }
        const data = await res.json();
        console.log("Debug: Fetched columns:", data);
        setColumns(Array.isArray(data.columns) ? data.columns : []);
        setSelectedTarget("");
        setTargetUniqueCount(null);
        setResult(null);
      } catch (err) {
        console.error("Failed to fetch columns", err);
        setColumns([]);
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
      if (!selectedDataset || !selectedTarget) {
        setTargetUniqueCount(null);
        return;
      }
      try {
        const res = await fetch(
          `/api/datasets/${selectedDataset}/column/${encodeURIComponent(
            selectedTarget
          )}/unique`,
          {
            method: "GET",
            credentials: "include",
          }
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
        console.log("Debug: Unique count:", data);
        setTargetUniqueCount(data.unique_count);
      } catch (err) {
        console.error("Failed to fetch unique count", err);
        setTargetUniqueCount(null);
      }
    };
    fetchUniqueCount();
  }, [selectedDataset, selectedTarget, navigate]);

  const isTargetValid =
    selectedModel === "PCA_KMeans" ||
    (selectedTarget && targetUniqueCount >= 2);

  const handleRunModel = async () => {
    if (!selectedDataset || !selectedModel) return;
    if (
      (selectedModel === "RandomForest" ||
        selectedModel === "LogisticRegression") &&
      !selectedTarget
    ) {
      setResult({
        error: "Please select a target column for classification models.",
      });
      return;
    }
    if (
      (selectedModel === "RandomForest" ||
        selectedModel === "LogisticRegression") &&
      targetUniqueCount < 2
    ) {
      setResult({
        error: `Target column '${selectedTarget}' has only ${targetUniqueCount} class(es). At least 2 are required.`,
      });
      return;
    }
    setIsLoading(true);
    try {
      const payload = {
        dataset_id: selectedDataset,
        model_name: selectedModel,
        n_clusters: nClusters,
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
      }
      console.log("Debug: Sending payload:", payload);
      const res = await fetch("/api/models/run", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      console.log("Debug: Model run response:", data);
      if (!res.ok) {
        console.error("🛑 POST /api/models/run failed:", res.status, data);
        setResult({ error: data.detail || `Model run failed (${res.status})` });
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error("🛑 Network error on /api/models/run:", err);
      setResult({ error: `Network error: ${err.message}` });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Run Pretrained Models</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-lg font-semibold mb-2">
            📁 Choose a Cleaned Dataset
          </h2>
          {datasets.length === 0 ? (
            <p className="text-gray-600 italic">
              No cleaned datasets available. Please upload and clean a dataset
              first.
            </p>
          ) : (
            <ul className="space-y-2">
              {datasets.map((ds) => (
                <li
                  key={ds.id}
                  className={`cursor-pointer p-2 rounded border hover:bg-blue-50 ${
                    selectedDataset === ds.id ? "bg-blue-100" : ""
                  }`}
                  onClick={() => setSelectedDataset(ds.id)}
                >
                  {ds.title}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <h2 className="text-lg font-semibold mb-2">⚙️ Choose a Model</h2>
          <ul className="space-y-2 mb-4">
            {models.map((model) => (
              <li
                key={model}
                className={`cursor-pointer p-2 rounded border hover:bg-green-50 ${
                  selectedModel === model ? "bg-green-100" : ""
                }`}
                onClick={() => setSelectedModel(model)}
              >
                {model}
              </li>
            ))}
          </ul>
          {(selectedModel === "RandomForest" ||
            selectedModel === "LogisticRegression") &&
            columns.length > 0 && (
              <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow mb-4">
                <label
                  htmlFor="target"
                  className="block text-sm font-semibold text-yellow-800"
                >
                  🎯 Target Column
                </label>
                <select
                  id="target"
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  className="mt-1 block w-full rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
                >
                  <option value="">Select a target column</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                {selectedTarget &&
                  targetUniqueCount !== null &&
                  targetUniqueCount < 2 && (
                    <p className="mt-2 text-red-600 text-sm">
                      ⚠️ Target column has only {targetUniqueCount} class(es).
                      At least 2 are required.
                    </p>
                  )}
              </div>
            )}
          {selectedModel === "RandomForest" && (
            <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow mb-4">
              <div className="mb-2">
                <label
                  htmlFor="n_estimators"
                  className="block text-sm font-semibold text-yellow-800"
                >
                  🌳 Number of Trees
                </label>
                <input
                  id="n_estimators"
                  type="number"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(parseInt(e.target.value))}
                  min="10"
                  step="10"
                  className="mt-1 block w-32 rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
                />
              </div>
              <div>
                <label
                  htmlFor="max_depth"
                  className="block text-sm font-semibold text-yellow-800"
                >
                  🌲 Max Depth (optional)
                </label>
                <input
                  id="max_depth"
                  type="number"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(e.target.value)}
                  min="1"
                  placeholder="None"
                  className="mt-1 block w-32 rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
                />
              </div>
            </div>
          )}
          {selectedModel === "LogisticRegression" && (
            <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow mb-4">
              <label
                htmlFor="C"
                className="block text-sm font-semibold text-yellow-800"
              >
                📈 Regularization Strength (C)
              </label>
              <input
                id="C"
                type="number"
                value={C}
                onChange={(e) => setC(parseFloat(e.target.value))}
                min="0.01"
                step="0.01"
                className="mt-1 block w-32 rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
              />
            </div>
          )}
          {selectedModel === "PCA_KMeans" && (
            <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow">
              <label
                htmlFor="clusters"
                className="block text-sm font-semibold text-yellow-800"
              >
                🎯 Number of Clusters
              </label>
              <select
                id="clusters"
                value={nClusters}
                onChange={(e) => setNClusters(parseInt(e.target.value))}
                className="mt-1 block w-32 rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
              >
                {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((val) => (
                  <option key={val} value={val}>
                    {val}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>
      <div className="mt-6">
        <button
          onClick={handleRunModel}
          className="bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-600"
          disabled={
            !selectedDataset || !selectedModel || isLoading || !isTargetValid
          }
        >
          {isLoading ? "Running..." : "Run Model"}
        </button>
      </div>
      {result && (
        <div className="mt-6 bg-gray-100 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4">🧾 Model Output</h2>
          {result.error && (
            <p className="text-red-600 mb-4 border border-red-300 bg-red-50 p-3 rounded">
              ❌ Error: {result.error}
              {(result.error.includes("preprocess the dataset") ||
                result.error.includes("NoSuchKey") ||
                result.error.includes("Missing target_column") ||
                result.error.includes("No valid numeric feature columns") ||
                result.error.includes("Empty dataset")) && (
                <span className="block mt-2 text-sm">
                  {result.error.includes("NoSuchKey")
                    ? "The cleaned dataset file is missing. Please re-clean the dataset in the Data Cleaning page or upload a new file."
                    : result.error.includes("preprocess the dataset")
                    ? "The dataset has missing values (e.g., columns with all NaNs). Use the Data Cleaning page to drop or impute these values."
                    : result.error.includes("Missing target_column")
                    ? "Please select a valid target column for this model."
                    : result.error.includes("Empty dataset")
                    ? "The dataset is empty. Please upload a valid dataset or check your cleaning steps."
                    : "The dataset lacks sufficient numeric features. Try cleaning or transforming the data in the Data Cleaning page."}{" "}
                  Ask the chatbot for guidance on preparing your dataset!
                </span>
              )}
            </p>
          )}
          {result.image_base64 ? (
            <>
              {console.log(
                "Debug: image_base64 length:",
                result.image_base64.length,
                "first chars:",
                result.image_base64.slice(0, 20)
              )}
              <img
                src={`data:image/png;base64,${result.image_base64}`}
                alt="Model result"
                className="w-full max-w-lg mx-auto rounded-lg shadow-md mb-4"
                onError={() =>
                  console.error("Debug: Failed to load plot image")
                }
                onLoad={() =>
                  console.log("Debug: Plot image loaded successfully")
                }
              />
            </>
          ) : (
            !result.error && (
              <p className="text-gray-600">
                No plot available. Ensure the model ran successfully or check
                the console for errors.
              </p>
            )
          )}
          {result.n_clusters && (
            <p className="mb-2">
              <strong>Clusters:</strong> {result.n_clusters}
            </p>
          )}
          {result.cluster_counts && (
            <div className="mb-4">
              <h3 className="font-medium">Cluster Counts:</h3>
              <ul className="list-disc list-inside">
                {Object.entries(result.cluster_counts).map(
                  ([cluster, count]) => (
                    <li key={cluster}>
                      Cluster {cluster}: {count}
                    </li>
                  )
                )}
              </ul>
            </div>
          )}
          {result.pca_variance_ratio && (
            <div className="mb-4">
              <h3 className="font-medium">Explained Variance (PCA):</h3>
              <ul className="list-disc list-inside">
                {result.pca_variance_ratio.map((val, i) => (
                  <li key={i}>
                    PC{i + 1}: {(val * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.message && (
            <p className="mt-2 text-sm text-gray-600">✅ {result.message}</p>
          )}
        </div>
      )}
    </div>
  );
}
