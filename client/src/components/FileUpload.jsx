import React, { useState } from "react";

export default function FileUpload() {
  /* ---------------- state ---------------- */
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);
  const [insights, setInsights] = useState(null); // New state for full insights
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [s3Key, setS3Key] = useState(null);
  const [records, setRecords] = useState(null);
  /* ------------- helpers ----------------- */
  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] ?? null);
  };

  /* ------------- upload ------------------ */
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please sign in again.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/upload-csv", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!res.ok) {
        if (res.status === 401) {
          throw new Error("Not authenticated – please log in again.");
        }
        const errPayload = await res.json().catch(() => ({}));
        throw new Error(errPayload.detail ?? "Upload failed");
      }

      const data = await res.json();
      console.log("Data returned from backend:", data);
      setInsights(data);
      setPreview(data.preview);
      setRecords(data.records);
      setS3Key(data.s3_key);
      setError(null);
    } catch (err) {
      console.error(err);
      setError(err.message);
    }
  };

  /* ------------- save dataset ------------ */
  const handleSave = async (e) => {
    e.preventDefault();
    if (!preview || !file) return;

    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please sign in again.");
      return;
    }

    try {
      const payload = {
        title,
        description,
        filename: file.name,
        raw_data: records,
        s3_key: s3Key,
      };
      console.log("Save payload:", payload);

      const res = await fetch("/datasets/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errPayload = await res.json().catch(() => ({}));
        throw new Error(errPayload.detail ?? "Save failed");
      }

      const data = await res.json();
      setSuccess(`Dataset saved! ID: ${data.id}`);
      setError(null);
    } catch (err) {
      console.error("Save error:", err);
      setError(err.message);
      setSuccess(null);
    }
  };

  return (
    <div className="bg-cyan-50 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h2 className="text-xl font-semibold mb-4">Upload CSV File</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm p-2"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm p-2"
            rows={3}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">CSV File</label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="mt-1 block w-full text-sm text-gray-600"
            required
          />
        </div>

        <button
          type="submit"
          className="w-full bg-blue-800 text-white py-2 px-4 rounded-md hover:bg-indigo-500"
        >
          Upload and Preview
        </button>
      </form>

      {error && <p className="text-red-600 mt-4">{error}</p>}
      {success && <p className="text-green-600 mt-4">{success}</p>}

      {insights && (
        <div className="mt-6 space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-700">Dataset Summary</h3>
            <p><strong>Shape:</strong> {insights.shape?.[0]} rows × {insights.shape?.[1]} columns</p>
            <p><strong>Columns:</strong> {insights.columns?.join(", ")}</p>
          </div>

          <div>
            <h3 className="font-semibold text-gray-700">Preview Rows</h3>
            <div className="overflow-auto max-h-64 border rounded">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-100 sticky top-0">
                  <tr>
                    {Object.keys(insights.preview?.[0] || {}).map((col) => (
                      <th key={col} className="px-2 py-1 border">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {insights.preview?.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="px-2 py-1 border">{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-700">Data Types</h3>
            <ul className="list-disc list-inside text-sm">
              {Object.entries(insights.dtypes || {}).map(([col, dtype]) => (
                <li key={col}><strong>{col}</strong>: {dtype}</li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-700">Missing Values</h3>
            <ul className="list-disc list-inside text-sm">
              {Object.entries(insights.null_counts || {}).map(([col, count]) => (
                <li key={col}><strong>{col}</strong>: {count}</li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-700">df.info()</h3>
            <pre className="overflow-auto max-h-64 bg-gray-100 p-4 rounded text-xs whitespace-pre-wrap">
              {insights.info_output}
            </pre>
          </div>

          <form onSubmit={handleSave} className="pt-4 border-t">
            <h3 className="text-md font-semibold text-gray-800 mb-2">Save to Database</h3>
            <button
              type="submit"
              className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-500"
            >
              Save to Database
            </button>
          </form>
        </div>
      )}
    </div>
  );
}




