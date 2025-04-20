import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

export default function DatasetDetail() {
  const { id } = useParams();
  const [dataset, setDataset] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:8000/datasets/${id}`)
      .then((res) => res.json())
      .then(setDataset)
      .catch(console.error);
  }, [id]);

  const fetchHeatmap = async () => {
    const res = await fetch(`http://localhost:8000/datasets/${id}/heatmap`);
    const data = await res.json();
    setHeatmapUrl(data.plot);
  };

  if (!dataset) return <div>Loading...</div>;

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">{dataset.title}</h2>
      <p className="text-gray-600 mb-2">{dataset.description}</p>
      <p className="text-sm text-gray-400 mb-4">
        Uploaded: {new Date(dataset.uploaded_at).toLocaleString()}
      </p>

      <div className="overflow-auto text-sm bg-gray-100 p-4 rounded">
        <h3 className="font-semibold mb-2">Raw Data Preview</h3>
        <table className="min-w-full divide-y divide-gray-300 text-xs">
          <thead className="bg-gray-200 text-gray-700">
            <tr>
              {Object.keys(dataset.raw_data[0]).map((key) => (
                <th key={key} className="px-2 py-1 text-left font-medium">
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-300">
            {dataset.raw_data.slice(0, 5).map((row, rowIndex) => (
              <tr key={rowIndex}>
                {Object.values(row).map((val, colIndex) => (
                  <td key={colIndex} className="px-2 py-1 whitespace-nowrap">
                    {val}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button
        onClick={fetchHeatmap}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500"
      >
        View Correlation Heatmap
      </button>

      {heatmapUrl && (
        <img
          src={heatmapUrl}
          alt="Correlation Heatmap"
          className="mt-4 rounded shadow-lg"
        />
      )}
    </div>
  );
}
