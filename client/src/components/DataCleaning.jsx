import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function DataCleaning() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [previewData, setPreviewData] = useState([]);
  const [cleanedData, setCleanedData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [columnMetadata, setColumnMetadata] = useState(null);
  const [nRows, setNRows] = useState(0);
  const [beforeStats, setBeforeStats] = useState(null);
  const [afterStats, setAfterStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [selectedSuggestion, setSelectedSuggestion] = useState("");
  const [options, setOptions] = useState({
    fillna_strategy: "",
    scale: "",
    encoding: "",
    lowercase_headers: false,
    dropna: false,
    remove_duplicates: false,
    outlier_method: "",
    conversions: {},
    binning: {},
    selected_columns: { fillna: [], scale: [], encoding: [], outliers: [] },
  });
  const [filename, setFilename] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [visImage, setVisImage] = useState(null);

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
        console.log("API Response:", data);
        setPreviewData(data.preview_data || []);
        setColumns(
          data.columns ||
            (data.preview_data?.[0] ? Object.keys(data.preview_data[0]) : [])
        );
        setColumnMetadata(data.column_metadata || null);
        setNRows(data.n_rows || 0);
        setFilename(data.filename || `dataset_${id}`);
      } catch (err) {
        console.error("fetchDataset error:", err);
        setError(err.message || "Failed to load dataset");
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [id, navigate]);

  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        const backendUrl =
          process.env.NODE_ENV === "development"
            ? "http://127.0.0.1:8000"
            : process.env.NORTHFLANK_GPU_URL || "";

        const res = await fetch(
          `${backendUrl}/api/databot/suggestions/${id}?page=data-cleaning`,
          {
            credentials: "include",
          }
        );

        if (res.status === 404) {
          setAlerts((prev) => [
            ...new Set([
              ...prev,
              "Databot suggestions not available for this dataset.",
            ]),
          ]);
          return;
        }

        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

        const data = await res.json();
        setAlerts((prev) => [...new Set([...prev, ...data.suggestions])]);
      } catch (err) {
        console.error("Failed to fetch Databot suggestions:", err);
        setAlerts((prev) => [
          ...new Set([
            ...prev,
            "Failed to fetch Databot suggestions: " + err.message,
          ]),
        ]);
      }
    };
    fetchSuggestions();
  }, [id]);

  const handleColumnSelect = (operation, column, checked) => {
    logAction(`Selected column '${column}' for ${operation}: ${checked}`);
    setOptions((prev) => ({
      ...prev,
      selected_columns: {
        ...prev.selected_columns,
        [operation]: checked
          ? [...prev.selected_columns[operation], column]
          : prev.selected_columns[operation].filter((c) => c !== column),
      },
    }));
  };

  const logAction = (action) => {
    fetch(`/api/databot/track/${id}`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    }).catch((err) => console.error("Failed to log action:", err));
  };

  const handleConversion = (column, type) => {
    setOptions((prev) => ({
      ...prev,
      conversions: { ...prev.conversions, [column]: type || undefined },
    }));
  };

  const handleBinning = (column, bins) => {
    setOptions((prev) => ({
      ...prev,
      binning: { ...prev.binning, [column]: parseInt(bins) || undefined },
    }));
  };

  alerts;

  const handlePreview = async () => {
    setLoading(true);
    setAlerts([]);
    setVisImage(null);
    try {
      console.log("State payload:", { dataset_id: Number(id), options });
      await fetch(`/api/databot/state/${id}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: Number(id), options }),
      });
      const dataList = [
        ...options.selected_columns.fillna.map((col) => ({
          column: col,
          operation: "fillna",
          value: options.fillna_strategy,
        })),
        ...Object.entries(options.conversions).map(([col, type]) => ({
          column: col,
          operation: "convert",
          value: type,
        })),
        ...options.selected_columns.scale.map((col) => ({
          column: col,
          operation: "scale",
          value: options.scale,
        })),
        ...options.selected_columns.encoding.map((col) => ({
          column: col,
          operation: "encoding",
          value: options.encoding,
        })),
        ...options.selected_columns.outliers.map((col) => ({
          column: col,
          operation: "outliers",
          value: options.outlier_method,
        })),
        ...Object.entries(options.binning).map(([col, bins]) => ({
          column: col,
          operation: "binning",
          value: bins,
        })),
      ].filter((op) => op.value);
      console.log("Clean payload:", {
        dataset_id: Number(id),
        data: dataList,
        operations: options,
      });
      const res = await fetch(`/api/datasets/${id}/clean-preview`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: Number(id),
          data: dataList,
          operations: options,
          save: false, // Preview only
        }),
      });
      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error((await res.text()) || "Preview failed");
      const data = await res.json();
      setBeforeStats(data.before_stats);
      setAfterStats(data.after_stats);
      setAlerts(data.alerts || []);
      setCleanedData(data.preview || previewData.slice(0, 10));
      setVisImage(data.vis_image_base64 || null);
    } catch (err) {
      setError(err.message || "Preview failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setAlerts([]);

    try {
      // Map current options into clean and preprocess
      const clean = {
        dropna: options.dropna,
        fillna_strategy: options.fillna_strategy,
        lowercase_headers: options.lowercase_headers,
        remove_duplicates: options.remove_duplicates,
      };

      const preprocess = {
        scale: options.scale,
        encoding: options.encoding,
      };

      const res = await fetch(`/api/datasets/${id}/process`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clean,
          preprocess,
        }),
      });

      if (res.status === 401) return navigate("/login");
      if (!res.ok) throw new Error((await res.text()) || "Save failed");

      const data = await res.json();
      setAlerts(data.alerts || []);
      if (data.saved) navigate(`/datasets/${id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const renderStats = (stats, title) => (
    <div className="bg-gray-100 p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm">
        <strong>Rows × Columns:</strong> {stats.shape.join(" × ")}
      </p>
      <div className="mt-2">
        <p className="font-medium text-sm">Null Counts:</p>
        <table className="w-full text-xs">
          <tbody>
            {Object.entries(stats.null_counts).map(([key, val]) => (
              <tr key={key}>
                <td className="border px-2 py-1">{key}</td>
                <td className="border px-2 py-1">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2">
        <p className="font-medium text-sm">Data Types:</p>
        <table className="w-full text-xs">
          <tbody>
            {Object.entries(stats.dtypes).map(([key, val]) => (
              <tr key={key}>
                <td className="border px-2 py-1">{key}</td>
                <td className="border px-2 py-1">{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
  const renderPreviewTable = (data, title) => (
    <div className="mt-4">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div className="overflow-x-auto bg-gray-100 p-4 rounded-lg shadow">
        <table className="w-full text-xs">
          <thead>
            <tr>
              {data.length > 0 &&
                Object.keys(data[0]).map((col) => (
                  <th key={col} className="border px-2 py-1 font-medium">
                    {col}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 10).map((row, i) => (
              <tr key={i} className="hover:bg-gray-200">
                {Object.values(row).map((val, j) => (
                  <td key={j} className="border px-2 py-1">
                    {val === null || val === undefined ? "N/A" : String(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 10 && (
          <p className="mt-2 text-xs text-gray-600">Showing first 10 rows.</p>
        )}
      </div>
    </div>
  );

  const getInfoOutput = () => {
    if (
      !columnMetadata ||
      Object.keys(columnMetadata).length === 0 ||
      columns.length === 0
    ) {
      return "No dataset information available";
    }
    const dtypeCounts = Object.values(columnMetadata).reduce((acc, meta) => {
      const dtype = meta.dtype || "unknown";
      acc[dtype] = (acc[dtype] || 0) + 1;
      return acc;
    }, {});
    return `<class 'pandas.core.frame.DataFrame'>
RangeIndex: ${nRows} entries, 0 to ${nRows - 1}
Data columns (total ${columns.length} columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
${columns
  .map((name, index) => {
    const meta = columnMetadata[name] || {};
    const nonNullCount =
      meta.null_count !== undefined ? nRows - meta.null_count : "Unknown";
    const dtype = meta.dtype || "unknown";
    return ` ${index.toString().padStart(2, " ")}  ${name.padEnd(
      30
    )} ${nonNullCount.toString().padEnd(15)} ${dtype}`;
  })
  .join("\n")}
dtypes: ${Object.entries(dtypeCounts)
      .map(([dtype, count]) => `${dtype}(${count})`)
      .join(", ")}
memory usage: Unknown`;
  };

  if (loading)
    return (
      <div className="p-4 text-center text-lg text-gray-600">Loading...</div>
    );
  if (error)
    return <div className="p-4 text-center text-lg text-red-500">{error}</div>;

  return (
    <div className="max-w-6xl mx-auto p-4 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 flex items-center">
        <svg
          className="w-6 h-6 mr-2 text-blue-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-6l-2-2H5a2 2 0 0 0-2 2z"
          />
        </svg>
        Clean Dataset: {filename}
      </h2>

      {/* Side-by-Side Dataset Peek and Alerts & Suggestions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Dataset Peek Dropdown */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Dataset Peek</h3>
          <details className="text-xs bg-white border border-gray-300 rounded">
            <summary className="p-2 cursor-pointer bg-gray-100 hover:bg-gray-200">
              Dataset Info
            </summary>
            <pre className="p-2 whitespace-pre-wrap">{getInfoOutput()}</pre>
          </details>
        </div>

        {/* Alerts & Suggestions Dropdown */}
        {alerts.length > 0 && (
          <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-500">
            <h3 className="font-semibold text-yellow-800 mb-2">
              Alerts & Suggestions
            </h3>
            <select
              className="w-full rounded border-gray-300 px-3 py-2 text-sm text-yellow-700 bg-yellow-50 focus:outline-none focus:ring-2 focus:ring-yellow-500"
              title="View Databot suggestions for cleaning the dataset"
              value={selectedSuggestion}
              onChange={(e) => setSelectedSuggestion(e.target.value)}
            >
              <option value="">Select a suggestion</option>
              {alerts.map((alert, index) => (
                <option key={index} value={alert}>
                  {alert}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Imputation */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">Missing Values</h4>
          <select
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                fillna_strategy: e.target.value,
              }))
            }
            value={options.fillna_strategy}
            className="w-full rounded border-gray-300 px-3 py-1 mb-2"
            title="Choose how to fill missing values in selected columns."
          >
            <option value="">No Imputation</option>
            <option value="mean">Mean (numeric)</option>
            <option value="median">Median (numeric)</option>
            <option value="mode">Mode (categorical)</option>
            <option value="zero">Zero</option>
            <option value="knn">KNN (numeric)</option>
          </select>
          <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
            {columns.map((col) => (
              <label key={col} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={options.selected_columns.fillna.includes(col)}
                  onChange={(e) =>
                    handleColumnSelect("fillna", col, e.target.checked)
                  }
                  className="mr-1"
                />
                {col}
              </label>
            ))}
          </div>
        </div>

        {/* Scaling */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">Scale Numeric Columns</h4>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, scale: e.target.value }))
            }
            value={options.scale}
            className="w-full rounded border-gray-300 px-3 py-1 mb-2"
            title="Scale numeric columns to a common range."
          >
            <option value="">No Scaling</option>
            <option value="normalize">Min-Max (0-1)</option>
            <option value="standardize">Z-Score</option>
            <option value="robust">Robust (IQR)</option>
          </select>
          <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
            {columns.map((col) => (
              <label key={col} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={options.selected_columns.scale.includes(col)}
                  onChange={(e) =>
                    handleColumnSelect("scale", col, e.target.checked)
                  }
                  className="mr-1"
                />
                {col}
              </label>
            ))}
          </div>
        </div>

        {/* Encoding */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">
            Encode Categorical Columns
          </h4>
          <select
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, encoding: e.target.value }))
            }
            value={options.encoding}
            className="w-full rounded border-gray-300 px-3 py-1 mb-2"
            title="Convert categorical columns to numeric format."
          >
            <option value="">No Encoding</option>
            <option value="onehot">One-Hot</option>
            <option value="label">Label</option>
            <option value="ordinal">Ordinal</option>
          </select>
          <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
            {columns.map((col) => (
              <label key={col} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={options.selected_columns.encoding.includes(col)}
                  onChange={(e) =>
                    handleColumnSelect("encoding", col, e.target.checked)
                  }
                  className="mr-1"
                />
                {col}
              </label>
            ))}
          </div>
        </div>

        {/* Outliers */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">Handle Outliers</h4>
          <select
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                outlier_method: e.target.value,
              }))
            }
            value={options.outlier_method}
            className="w-full rounded border-gray-300 px-3 py-1 mb-2"
            title="Remove or cap outliers in numeric columns."
          >
            <option value="">No Handling</option>
            <option value="iqr">IQR Removal</option>
            <option value="zscore">Z-Score Removal</option>
            <option value="cap">Cap at Percentiles</option>
          </select>
          <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
            {columns.map((col) => (
              <label key={col} className="flex items-center text-sm">
                <input
                  type="checkbox"
                  checked={options.selected_columns.outliers.includes(col)}
                  onChange={(e) =>
                    handleColumnSelect("outliers", col, e.target.checked)
                  }
                  className="mr-1"
                />
                {col}
              </label>
            ))}
          </div>
        </div>

        {/* Conversions */}
        <div className="col-span-2 bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">Convert Data Types</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {columns.map((col) => (
              <div key={col}>
                <label className="text-sm">{col}</label>
                <select
                  onChange={(e) => handleConversion(col, e.target.value)}
                  value={options.conversions[col] || ""}
                  className="w-full rounded border-gray-300 px-2 py-1 text-xs"
                  title={`Change the data type of '${col}'.`}
                >
                  <option value="">No Change</option>
                  <option value="numeric">Numeric</option>
                  <option value="date">Date</option>
                  <option value="category">Category</option>
                </select>
              </div>
            ))}
          </div>
        </div>

        {/* Binning */}
        <div className="col-span-2 bg-gray-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold mb-2">Bin Numeric Columns</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {columns.map((col) => (
              <div key={col}>
                <label className="text-sm">{col}</label>
                <input
                  type="number"
                  min="2"
                  placeholder="Bins"
                  value={options.binning[col] || ""}
                  onChange={(e) => handleBinning(col, e.target.value)}
                  className="w-full rounded border-gray-300 px-2 py-1 text-xs"
                  title={`Group '${col}' into discrete bins (e.g., price ranges).`}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-4 mb-4">
        <label className="flex items-center text-sm">
          <input
            type="checkbox"
            checked={options.lowercase_headers}
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                lowercase_headers: e.target.checked,
              }))
            }
            className="mr-1"
          />
          Lowercase Headers
        </label>
        <label className="flex items-center text-sm">
          <input
            type="checkbox"
            checked={options.dropna}
            onChange={(e) =>
              setOptions((prev) => ({ ...prev, dropna: e.target.checked }))
            }
            className="mr-1"
          />
          Drop NA Rows
        </label>
        <label className="flex items-center text-sm">
          <input
            type="checkbox"
            checked={options.remove_duplicates}
            onChange={(e) =>
              setOptions((prev) => ({
                ...prev,
                remove_duplicates: e.target.checked,
              }))
            }
            className="mr-1"
          />
          Remove Duplicates
        </label>
      </div>

      <div className="flex justify-center space-x-4 mb-4">
        <button
          onClick={handlePreview}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-500 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Previewing..." : "Preview Cleaning"}
        </button>
        {beforeStats && afterStats && (
          <button
            onClick={handleSave}
            disabled={saving || alerts.some((a) => a.includes("Failed"))}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-500 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save Cleaned Dataset"}
          </button>
        )}
      </div>

      {beforeStats && afterStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {renderStats(beforeStats, "Before Cleaning")}
          {renderStats(afterStats, "After Cleaning")}
        </div>
      )}

      {(previewData.length > 0 || cleanedData.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {previewData.length > 0 &&
            renderPreviewTable(previewData, "Original Data")}
          {cleanedData.length > 0 &&
            renderPreviewTable(cleanedData, "Cleaned Data")}
        </div>
      )}

      {visImage && (
        <div className="mt-4 text-center">
          <h3 className="text-lg font-semibold mb-2">Correlation Heatmap</h3>
          <img
            src={`data:image/png;base64,${visImage}`}
            alt="Correlation Heatmap"
            className="max-w-full rounded shadow mx-auto"
          />
        </div>
      )}
    </div>
  );
}
