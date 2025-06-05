// client/src/components/Models.jsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Models() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForest");
  const [result, setResult] = useState(null);

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
        setDatasets(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Failed to fetch cleaned datasets", err);
        setDatasets([]);
      }
    };

    fetchCleanedDatasets();
  }, [navigate]);

  const handleRunModel = async () => {
    if (!selectedDataset || !selectedModel) return;
    setIsLoading(true);
    try {
      const res = await fetch("/api/models/run", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_name: selectedModel,
          n_clusters: nClusters,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("🛑 POST /api/models/run failed:", res.status, errorText);
        setResult({ error: `Model run failed (${res.status}): ${errorText}` });
      } else {
        const data = await res.json();
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
          <h2 className="text-lg font-semibold mb-2">📁 Choose a Dataset</h2>
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
          disabled={!selectedDataset || !selectedModel || isLoading}
        >
          {isLoading ? "Running..." : "Run Model"}
        </button>
      </div>

      {result && (
        <div className="mt-6 bg-gray-100 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4">🧾 Model Output</h2>

          {result.image_base64 && (
            <img
              src={`data:image/png;base64,${result.image_base64}`}
              alt="Model result"
              className="w-full max-w-lg mx-auto rounded-lg shadow-md mb-4"
            />
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
