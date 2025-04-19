import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

export default function DatasetDetail() {
  const { id } = useParams();
  const [dataset, setDataset] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:8000/datasets/${id}`)
      .then((res) => res.json())
      .then(setDataset)
      .catch(console.error);
  }, [id]);

  if (!dataset) return <div>Loading...</div>;

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">{dataset.title}</h2>
      <p className="text-gray-600 mb-2">{dataset.description}</p>
      <p className="text-sm text-gray-400 mb-4">Uploaded: {new Date(dataset.uploaded_at).toLocaleString()}</p>
      <div className="overflow-auto text-sm bg-gray-100 p-4 rounded">
        <h3 className="font-semibold mb-2">Raw Data Preview</h3>
        <pre>{JSON.stringify(dataset.raw_data?.slice?.(0, 5), null, 2)}</pre>
      </div>
    </div>
  );
}
