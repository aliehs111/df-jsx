import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";

export default function DatasetDetail() {
  const { id } = useParams();
  const [dataset, setDataset] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      // kick them back to login if somehow unauthenticated
      navigate("/");
      return;
    }

    fetch(`/datasets/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => {
        if (!res.ok) throw new Error("Could not load dataset");
        return res.json();
      })
      .then(setDataset)
      .catch(err => {
        console.error(err);
        // optional: show toast or redirect
      });
  }, [id, navigate]);

  const fetchHeatmap = async () => {
    const token = localStorage.getItem("token");
    const res = await fetch(
      `/datasets/${id}/heatmap`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (!res.ok) {
      alert("Could not generate heat-map");
      return;
    }
    const data = await res.json();
    setHeatmapUrl(data.plot);
  };
  

  if (!dataset) return <div className="p-6">Loading…</div>;

  /* ─────────────────────────────────── RENDER ────────────────────────────── */
  if (!dataset) return <div className="p-4">Loading…</div>;

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-bold mb-4">{dataset.title}</h2>
      <p className="text-gray-600 mb-2">{dataset.description}</p>
      <p className="text-sm text-gray-400 mb-4">
        Uploaded: {new Date(dataset.uploaded_at).toLocaleString()}
      </p>

      {/* preview (first 5 rows) */}
      <div className="overflow-auto text-sm bg-gray-100 p-4 rounded">
        <h3 className="font-semibold mb-2">Raw Data Preview</h3>
        <table className="min-w-full divide-y divide-gray-300 text-xs">
          <thead className="bg-gray-200">
            <tr>
              {Object.keys(dataset.raw_data[0]).map((k) => (
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

      <button
        onClick={fetchHeatmap}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500"
      >
        View Correlation Heat-map
      </button>

      {heatmapUrl && (
        <img
          src={heatmapUrl}
          alt="Correlation heat-map"
          className="mt-4 rounded shadow-lg"
        />
      )}

      <Link
        to={`/datasets/${id}/clean`}
        className="inline-block mt-4 text-blue-600 hover:underline font-medium"
      >
        Begin Cleaning &amp; Wrangling
      </Link>
    </div>
  );
}

