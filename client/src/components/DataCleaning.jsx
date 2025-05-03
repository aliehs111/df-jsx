import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

export default function DataCleaning() {
  const { id } = useParams();
  const [rawData, setRawData] = useState([]);
  const [cleanedData, setCleanedData] = useState([]);
  const [beforeStats, setBeforeStats] = useState(null);
  const [afterStats, setAfterStats] = useState(null);
  const [options, setOptions] = useState({});

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
  
    fetch(`/datasets/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Cannot load dataset");
        return res.json();
      })
      .then((data) => setRawData(data.raw_data))
      .catch(console.error);
  }, [id]);
  

  const handlePreview = async () => {
    const token = localStorage.getItem("token");
    if (!token) return alert("Please log in again");
  
    const res = await fetch("/clean-preview", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ dataset_id: id, operations: options }),
    });
  
    if (!res.ok) {
      alert("Preview failed");
      return;
    }
  
    const data = await res.json();
    setBeforeStats(data.before_stats);
    setAfterStats(data.after_stats);
    // NOTE: backend returns no “preview” key – you probably want after_stats
    // setCleanedData(data.preview);
  }
  

  const renderStats = (stats) => (
    <div className="bg-gray-50 p-4 rounded shadow text-sm">
      <p><strong>Shape:</strong> {stats.shape.join(" x ")}</p>
      <div className="mt-2">
        <p className="font-semibold">Null Counts:</p>
        <table className="table-auto text-sm w-full mt-1">
          <tbody>
            {Object.entries(stats.null_counts).map(([key, val]) => (
              <tr key={key}>
                <td className="border px-2 py-1 w-1/2 font-medium">{key}</td>
                <td className="border px-2 py-1">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4">
        <p className="font-semibold">Data Types:</p>
        <table className="table-auto text-sm w-full mt-1">
          <tbody>
            {Object.entries(stats.dtypes).map(([key, val]) => (
              <tr key={key}>
                <td className="border px-2 py-1 w-1/2 font-medium">{key}</td>
                <td className="border px-2 py-1">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white shadow rounded">
      <h2 className="text-2xl font-bold mb-4">Quick Cleaning Pipeline</h2>

      <div className="grid gap-4 mb-6">
        <div>
          <label className="block font-medium">Missing Value Strategy</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, fillna_strategy: e.target.value }))
            }
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="mean">Fill with Mean</option>
            <option value="median">Fill with Median</option>
            <option value="mode">Fill with Mode</option>
            <option value="zero">Fill with 0</option>
          </select>
        </div>

        <div>
          <label className="block font-medium">Scaling</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, scale: e.target.value }))
            }
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="normalize">Normalize (Min-Max)</option>
            <option value="standardize">Standardize (Z-score)</option>
          </select>
        </div>

        <div>
          <label className="block font-medium">Categorical Encoding</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, encoding: e.target.value }))
            }
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="onehot">One-Hot Encoding</option>
            <option value="label">Label Encoding</option>
          </select>
        </div>

        <div>
          <label className="inline-flex items-center">
            <input
              type="checkbox"
              className="mr-2"
              onChange={(e) =>
                setOptions((prev) => ({ ...prev, lowercase_headers: e.target.checked }))
              }
            />
            Convert column names to lowercase
          </label>
        </div>
      </div>

      <button
        onClick={handlePreview}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500"
      >
        Preview Cleaning
      </button>

      {beforeStats && afterStats && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-xl font-semibold mb-2">Before Stats</h3>
            {renderStats(beforeStats)}
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-2">After Stats</h3>
            {renderStats(afterStats)}
          </div>
        </div>
      )}

{Array.isArray(cleanedData) && cleanedData.length > 0 && (

        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-2">Cleaned Data Preview</h3>
          <div className="overflow-auto bg-gray-50 p-4 rounded">
            <table className="table-auto w-full text-sm text-left text-gray-700">
              <thead>
                <tr>
                  {Object.keys(cleanedData[0]).map((col) => (
                    <th key={col} className="border px-2 py-1 font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cleanedData.slice(0, 5).map((row, i) => (
                  <tr key={i} className="hover:bg-gray-100">
                    {Object.values(row).map((val, j) => (
                      <td key={j} className="border px-2 py-1">
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
    </div>
  );
}




