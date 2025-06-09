import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function DataCleaning() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [rawData, setRawData] = useState([]);
  const [cleanedData, setCleanedData] = useState([]);
  const [beforeStats, setBeforeStats] = useState(null);
  const [afterStats, setAfterStats] = useState(null);
  const [options, setOptions] = useState({});
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const res = await fetch(`/api/datasets/${id}`, {
          credentials: "include",
        });
        if (res.status === 401) return navigate("/login");
        if (res.status === 404) return navigate("/datasets");
        if (!res.ok) throw new Error("Cannot load dataset");
        const data = await res.json();
        setRawData(data.raw_data);
        setFilename(data.filename);
      } catch (err) {
        setError(err.message || "Failed to load dataset");
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [id, navigate]);

  const handlePreview = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/datasets/${id}/clean-preview`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: Number(id), operations: options }),
      });
      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error("Preview failed");
      const data = await res.json();
      setBeforeStats(data.before_stats);
      setAfterStats(data.after_stats);
      setCleanedData(data.preview || []);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const payload = {
        clean: {
          dropna: options.dropna || false,
          fillna_strategy: options.fillna_strategy || "",
          lowercase_headers: options.lowercase_headers || false,
          remove_duplicates: options.remove_duplicates || false,
        },
        preprocess: {
          scale: options.scale || "",
          encoding: options.encoding || "",
        },
      };

      const res = await fetch(`/api/datasets/${id}/process`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.status === 401) {
        navigate("/login");
        return;
      }
      if (!res.ok) {
        const err = await res.text().catch(() => "Save failed");
        throw new Error(err);
      }

      const { id: newId } = await res.json();
      navigate(`/datasets/${newId}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const renderStats = (stats) => (
    <div className="bg-gray-50 p-4 rounded shadow text-sm">
      <p>
        <strong>Shape:</strong> {stats.shape.join(" × ")}
      </p>
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

  if (loading) return <div className="p-6">Loading…</div>;
  if (error) return <div className="p-6 text-red-500">{error}</div>;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white shadow rounded">
      <h2 className="text-2xl font-bold mb-4 text-blue-800">
        Pipeline Sandbox
      </h2>
      <p className="text-gray-700 mb-6">
        File: <span className="font-medium">{filename}</span>
      </p>

      {/* Options Panel */}
      <div className="grid gap-4 mb-6">
        {/* Missing Value Strategy */}
        <div>
          <label className="block font-medium">Missing Value Strategy</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                fillna_strategy: e.target.value,
              }))
            }
            value={options.fillna_strategy || ""}
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="mean">Mean</option>
            <option value="median">Median</option>
            <option value="mode">Mode</option>
            <option value="zero">Zero</option>
          </select>
        </div>
        {/* Scaling */}
        <div>
          <label className="block font-medium">Scaling</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, scale: e.target.value }))
            }
            value={options.scale || ""}
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="normalize">Min-Max</option>
            <option value="standardize">Z-score</option>
          </select>
        </div>
        {/* Encoding */}
        <div>
          <label className="block font-medium">Categorical Encoding</label>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, encoding: e.target.value }))
            }
            value={options.encoding || ""}
            className="mt-1 block w-full rounded border border-gray-300 px-3 py-2"
          >
            <option value="">-- Choose --</option>
            <option value="onehot">One-Hot</option>
            <option value="label">Label</option>
          </select>
        </div>
        {/* Lowercase Columns */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="lowercase_headers"
            checked={options.lowercase_headers || false}
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                lowercase_headers: e.target.checked,
              }))
            }
            className="mr-2"
          />
          <label htmlFor="lowercase_headers" className="font-medium">
            Lowercase Headers
          </label>
        </div>
        {/* Drop NA */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="dropna"
            checked={options.dropna || false}
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, dropna: e.target.checked }))
            }
            className="mr-2"
          />
          <label htmlFor="dropna" className="font-medium">
            Drop Rows with NA
          </label>
        </div>
        {/* Remove Duplicates */}
        <div className="flex items-center">
          <input
            type="checkbox"
            id="remove_duplicates"
            checked={options.remove_duplicates || false}
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                remove_duplicates: e.target.checked,
              }))
            }
            className="mr-2"
          />
          <label htmlFor="remove_duplicates" className="font-medium">
            Remove Duplicate Rows
          </label>
        </div>
      </div>

      {/* Preview & Apply Buttons */}
      <div className="flex space-x-4 mb-6">
        <button
          onClick={handlePreview}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500 disabled:opacity-50"
        >
          Preview Cleaning
        </button>
        {beforeStats && afterStats && (
          <button
            onClick={handleSave}
            disabled={saving}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-500 disabled:opacity-50"
          >
            {saving ? "Saving…" : "Save Cleaned CSV"}
          </button>
        )}
      </div>

      {/* Stats Comparison */}
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

      {/* Cleaned Data Preview */}
      {cleanedData.length > 0 && (
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
                        {val === null || val === undefined ? "" : String(val)}
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
