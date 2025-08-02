import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Models() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [result, setResult] = useState(null);
  const [columns, setColumns] = useState([]);
  const [stringColumns, setStringColumns] = useState([]); // For Sentiment text columns
  const [selectedTarget, setSelectedTarget] = useState("");
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState("");
  const [C, setC] = useState(1.0);
  const [targetUniqueCount, setTargetUniqueCount] = useState(null);
  const [models, setModels] = useState([]);

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
        console.log("Debug: Fetched available models:", data);
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
              "Failed to load dataset columns. Please ensure the dataset is properly cleaned in Data Cleaning.",
          });
          return;
        }
        const data = await res.json();
        console.log("Debug: Fetched columns:", data);
        setColumns(Array.isArray(data.columns) ? data.columns : []);
        // TODO: Backend should return dtypes; for now, assume all cols valid for Sentiment
        // Replace with API call to filter string columns if available
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
  }, [selectedDataset, selectedTarget, selectedModel, navigate]);

  const isTargetValid =
    selectedModel === "PCA_KMeans" ||
    selectedModel === "Sentiment" || // Sentiment requires text column, no unique count check
    (selectedTarget && targetUniqueCount >= 2);

  const handleRunModel = async () => {
    if (!selectedDataset || !selectedModel) return;

    // Validate required target column(s)
    if (
      (selectedModel === "RandomForest" ||
        selectedModel === "LogisticRegression" ||
        selectedModel === "Sentiment" ||
        selectedModel === "FeatureImportance") &&
      !selectedTarget
    ) {
      setResult({
        error: `Please select a ${
          selectedModel === "Sentiment" ? "text" : "target"
        } column for this model.`,
      });
      return;
    }

    // Extra check for classification models
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

    // TimeSeriesForecasting requires two columns
    if (selectedModel === "TimeSeriesForecasting") {
      if (!selectedTarget || !selectedTarget.includes("|")) {
        setResult({
          error:
            "Please select both a date column and a value column for forecasting.",
        });
        return;
      }
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
        // No target_column needed
      } else if (selectedModel === "TimeSeriesForecasting") {
        payload.target_column = selectedTarget; // formatted as "date|value"
      } else if (selectedModel === "FeatureImportance") {
        payload.target_column = selectedTarget;
      }

      console.log("Debug: Sending payload:", payload);

      const res = await fetch("/api/models/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const text = await res.text(); // always read as text first
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        console.error("Non-JSON response:", text.slice(0, 200));
        alert(
          "Backend returned an error page instead of JSON.\n" +
            text.slice(0, 200)
        );
        return;
      }

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
      <h1 className="text-3xl font-bold mb-6">Run Models</h1>
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
                key={model.name}
                className={`cursor-pointer p-2 rounded border hover:bg-green-50 ${
                  selectedModel === model.name ? "bg-green-100" : ""
                }`}
                onClick={() => setSelectedModel(model.name)}
                title={model.description || ""}
              >
                {model.name}
              </li>
            ))}
          </ul>
          {(selectedModel === "RandomForest" ||
            selectedModel === "LogisticRegression" ||
            selectedModel === "Sentiment") &&
            columns.length > 0 && (
              <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow mb-4">
                <label
                  htmlFor="target"
                  className="block text-sm font-semibold text-yellow-800"
                >
                  {selectedModel === "Sentiment"
                    ? "📝 Text Column"
                    : "🎯 Target Column"}
                </label>
                <select
                  id="target"
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  className="mt-1 block w-full rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
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
          {selectedModel === "TimeSeriesForecasting" && columns.length > 0 && (
            <div className="bg-amber-50 border border-yellow-300 p-4 rounded shadow mb-4">
              <label
                htmlFor="dateColumn"
                className="block text-sm font-semibold text-yellow-800 mb-1"
              >
                📅 Date Column
              </label>
              <select
                id="dateColumn"
                value={selectedTarget.split("|")[0] || ""}
                onChange={(e) => {
                  const valueCol = selectedTarget.split("|")[1] || "";
                  setSelectedTarget(`${e.target.value}|${valueCol}`);
                }}
                className="mt-1 block w-full rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm mb-4"
              >
                <option value="">Select a date column</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>

              <label
                htmlFor="valueColumn"
                className="block text-sm font-semibold text-yellow-800 mb-1"
              >
                📈 Value Column
              </label>
              <select
                id="valueColumn"
                value={selectedTarget.split("|")[1] || ""}
                onChange={(e) => {
                  const dateCol = selectedTarget.split("|")[0] || "";
                  setSelectedTarget(`${dateCol}|${e.target.value}`);
                }}
                className="mt-1 block w-full rounded-md border-yellow-300 shadow-sm focus:border-yellow-500 focus:ring-yellow-500 sm:text-sm"
              >
                <option value="">Select a value column</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              {selectedTarget && !selectedTarget.includes("|") && (
                <p className="mt-2 text-red-600 text-sm">
                  ⚠️ Please select both a date and a value column.
                </p>
              )}
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
                result.error.includes("Missing date_column|value_column") ||
                result.error.includes("No valid numeric feature columns") ||
                result.error.includes("Empty dataset") ||
                result.error.includes("No text data")) && (
                <span className="block mt-2 text-sm">
                  {result.error.includes("NoSuchKey")
                    ? "The cleaned dataset file is missing. Please re-clean the dataset in the Data Cleaning page or upload a new file."
                    : result.error.includes("preprocess the dataset")
                    ? "The dataset has missing values (e.g., columns with all NaNs). Use the Data Cleaning page to drop or impute these values."
                    : result.error.includes("Missing target_column")
                    ? "Please select a valid target column for this model."
                    : result.error.includes("Missing date_column|value_column")
                    ? "Please select both a Date column and a Value column for forecasting."
                    : result.error.includes("Empty dataset")
                    ? "The dataset is empty. Please upload a valid dataset or check your cleaning steps."
                    : result.error.includes("No text data")
                    ? "No valid text in this column. Please select a different text column or clean your data."
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
          {result.model === "PCA_KMeans" && (
            <>
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
            </>
          )}
          {result.model === "RandomForest" && (
            <>
              {result.class_counts && (
                <div className="mb-4">
                  <h3 className="font-medium">Class Counts:</h3>
                  <ul className="list-disc list-inside">
                    {Object.entries(result.class_counts).map(([cls, count]) => (
                      <li key={cls}>
                        Class {cls}: {count}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {result.classification_report && (
                <div className="mb-4">
                  <h3 className="font-medium">Classification Report:</h3>
                  <pre className="text-sm bg-gray-50 p-2 rounded">
                    {JSON.stringify(result.classification_report, null, 2)}
                  </pre>
                </div>
              )}
              {result.confusion_matrix && (
                <div className="mb-4">
                  <h3 className="font-medium">Confusion Matrix:</h3>
                  <pre className="text-sm bg-gray-50 p-2 rounded">
                    {JSON.stringify(result.confusion_matrix, null, 2)}
                  </pre>
                </div>
              )}
              {result.feature_importances && (
                <div className="mb-4">
                  <h3 className="font-medium">Feature Importances:</h3>
                  <ul className="list-disc list-inside">
                    {result.feature_importances.map((val, i) => (
                      <li key={i}>
                        Feature {i + 1}: {val.toFixed(4)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
          {result.model === "LogisticRegression" && (
            <>
              {result.class_counts && (
                <div className="mb-4">
                  <h3 className="font-medium">Class Counts:</h3>
                  <ul className="list-disc list-inside">
                    {Object.entries(result.class_counts).map(([cls, count]) => (
                      <li key={cls}>
                        Class {cls}: {count}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {result.classification_report && (
                <div className="mb-4">
                  <h3 className="font-medium">Classification Report:</h3>
                  <pre className="text-sm bg-gray-50 p-2 rounded">
                    {JSON.stringify(result.classification_report, null, 2)}
                  </pre>
                </div>
              )}
              {result.confusion_matrix && (
                <div className="mb-4">
                  <h3 className="font-medium">Confusion Matrix:</h3>
                  <pre className="text-sm bg-gray-50 p-2 rounded">
                    {JSON.stringify(result.confusion_matrix, null, 2)}
                  </pre>
                </div>
              )}
              {result.coefficients && (
                <div className="mb-4">
                  <h3 className="font-medium">Coefficients:</h3>
                  <pre className="text-sm bg-gray-50 p-2 rounded">
                    {JSON.stringify(result.coefficients, null, 2)}
                  </pre>
                </div>
              )}
            </>
          )}
          {result.model === "Sentiment" && (
            <>
              {result.num_texts && (
                <p className="mb-2">
                  <strong>Processed Texts:</strong> {result.num_texts}
                </p>
              )}
              {result.sentiment_counts && (
                <div className="mb-4">
                  <h3 className="font-medium">Sentiment Counts:</h3>
                  <ul className="list-disc list-inside">
                    {Object.entries(result.sentiment_counts).map(
                      ([label, count]) => (
                        <li key={label}>
                          {label}: {count}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              )}
              {result.sample_results && (
                <div className="mb-4 overflow-x-auto">
                  <h3 className="font-medium">Sample Results:</h3>
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th>Text</th>
                        <th>Sentiment</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.sample_results.map(
                        ([text, label, score], idx) => (
                          <tr key={idx}>
                            <td>{text.slice(0, 100)}...</td>
                            <td>{label}</td>
                            <td>{score.toFixed(2)}</td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {result.message && (
            <p className="mt-2 text-sm text-gray-600">✅ {result.message}</p>
          )}

          {/* 👇 Replace your old AnomalyDetection section with this */}
          {result.model === "AnomalyDetection" && (
            <div>
              <h3 className="font-medium">Detected Anomalies</h3>
              {result.anomalies?.length > 0 ? (
                <ul>
                  {result.anomalies.map((a, i) => (
                    <li key={i}>{JSON.stringify(a)}</li>
                  ))}
                </ul>
              ) : (
                <p>No anomalies detected.</p>
              )}
            </div>
          )}

          {/* 👇 Replace your old TimeSeriesForecasting section with this */}
          {result.model === "TimeSeriesForecasting" && (
            <div>
              <h3 className="font-medium">Forecast Results</h3>
              {Array.isArray(result.forecast) ? (
                <pre className="text-sm bg-gray-50 p-2 rounded">
                  {JSON.stringify(result.forecast.slice(0, 10), null, 2)}
                </pre>
              ) : (
                <p>No forecast results available.</p>
              )}
            </div>
          )}

          {/* 👇 Replace your old FeatureImportance section with this */}
          {result.model === "FeatureImportance" && (
            <div>
              <h3 className="font-medium">Feature Importances</h3>
              {Array.isArray(result.importances) ? (
                <ul>
                  {result.importances.map(([feature, score], i) => (
                    <li key={i}>
                      {feature}: {score.toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No feature importances calculated.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
