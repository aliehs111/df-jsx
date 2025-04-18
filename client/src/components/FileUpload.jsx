import React, { useState } from 'react';

export default function FileUpload() {
  const [previewRows, setPreviewRows] = useState([]);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/upload-csv', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        console.error('Upload failed:', res.statusText);
        return;
      }
      const data = await res.json();
      setPreviewRows(data.head || []);
    } catch (err) {
      console.error('Error uploading file:', err);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-lg font-semibold mb-2">Upload CSV</h2>
      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
      />

      {previewRows.length > 0 && (
        <div className="mt-4 overflow-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                {Object.keys(previewRows[0]).map((col) => (
                  <th key={col} className="border px-2 py-1 text-left">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {previewRows.map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((val, j) => (
                    <td key={j} className="border px-2 py-1">{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
