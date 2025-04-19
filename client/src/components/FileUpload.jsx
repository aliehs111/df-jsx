import React, { useState } from "react";

export default function FileUpload() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload-csv", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setPreview(data);
      setError(null);
    } catch (err) {
      setError("Upload failed");
    }
  };

  return (
    <div className="bg-white rounded-xl shadow p-6 max-w-2xl mx-auto">
      <h2 className="text-xl font-semibold mb-4">Upload CSV File</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm p-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 shadow-sm p-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            rows={3}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            CSV File
          </label>
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
          className="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-500"
        >
          Upload and Preview
        </button>
      </form>

      {error && <p className="text-red-600 mt-4">{error}</p>}

      {preview && (
        <div className="mt-6 text-sm text-gray-700 space-y-2">
          <p><strong>Shape:</strong> {preview.shape?.[0]} rows × {preview.shape?.[1]} columns</p>
          <p><strong>Columns:</strong> {preview.columns?.join(", ")}</p>
          <pre className="overflow-auto max-h-64 bg-gray-100 p-4 rounded text-xs whitespace-pre-wrap">
            {preview.info_output}
          </pre>
        </div>
      )}
    </div>
  );
}



