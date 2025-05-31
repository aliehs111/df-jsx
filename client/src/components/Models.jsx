// client/src/components/Models.jsx
import { useEffect, useState } from "react";
import axios from "axios";

export default function Models() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForest");
  const [result, setResult] = useState(null);

  const models = ["RandomForest", "PCA_KMeans", "LogisticRegression"];

  useEffect(() => {
    const token = localStorage.getItem("token");
    axios
      .get("/datasets", {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((res) => setDatasets(res.data))
      .catch((err) => console.error("Failed to fetch datasets", err));
  }, []);

  const runModel = async () => {
    const token = localStorage.getItem("token");
    try {
      const res = await axios.post(
        `/models/run`,
        {
          dataset_id: selectedDataset,
          model_name: selectedModel,
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      setResult(res.data);
    } catch (err) {
      console.error("Failed to run model", err);
      alert("Failed to run model");
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">🧠 Run Pretrained Models</h1>

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
          <ul className="space-y-2">
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
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={runModel}
          className="bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-600"
          disabled={!selectedDataset || !selectedModel}
        >
          Run Model
        </button>
      </div>

      {result && (
        <div className="mt-6 bg-gray-100 p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2">🧾 Model Output</h2>
          <pre className="whitespace-pre-wrap text-sm text-gray-800">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
