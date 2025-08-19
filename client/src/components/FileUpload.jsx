// client/src/components/FileUpload.jsx
import React, { useState, useMemo, useCallback, useRef } from "react";

export default function FileUpload() {
  /* ---------------- config ---------------- */
  const MAX_FILE_SIZE_MB = 50;
  const ACCEPTED_MIME = new Set(["text/csv", "application/vnd.ms-excel"]);

  /* ---------------- state ---------------- */
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState(null);

  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // server response
  const [insights, setInsights] = useState(null);
  const [s3Key, setS3Key] = useState(null);

  // alerts: keep upload/save separate so we can render in different places
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(null);

  /* ------------- bottom alert anchor ------------- */
  const bottomAlertRef = useRef(null);
  const pingBottomAlert = () => {
    // wait for state update to render, then scroll near the Save area
    requestAnimationFrame(() => {
      bottomAlertRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    });
  };

  /* ------------- derived ----------------- */
  const fileMeta = useMemo(() => {
    if (!file) return null;
    return {
      name: file.name,
      sizeMB: (file.size / (1024 * 1024)).toFixed(2),
      type: file.type || "text/csv",
    };
  }, [file]);

  const canUpload = !!file && !isUploading && !isSaving;
  const canSave =
    !!insights &&
    !!s3Key &&
    !!file &&
    !!title.trim() &&
    !!description.trim() &&
    !isUploading &&
    !isSaving;

  /* ------------- helpers ----------------- */
  const clearAllAlerts = () => {
    setUploadError(null);
    setUploadSuccess(null);
    setSaveError(null);
    setSaveSuccess(null);
  };

  const clearSaveAlerts = () => {
    setSaveError(null);
    setSaveSuccess(null);
  };

  const validateFile = (f) => {
    if (!f) return "Please select a file.";
    const isCSV =
      f.name.toLowerCase().endsWith(".csv") || ACCEPTED_MIME.has(f.type);
    if (!isCSV) return "Only .csv files are supported.";
    if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024)
      return `File is too large (>${MAX_FILE_SIZE_MB}MB).`;
    return null;
  };

  const handleFileSelect = (f) => {
    clearAllAlerts();
    const problem = validateFile(f);
    if (problem) {
      setFile(null);
      setUploadError(problem);
      return;
    }
    setFile(f);
  };

  const handleInputChange = (e) =>
    handleFileSelect(e.target.files?.[0] ?? null);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    handleFileSelect(f ?? null);
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  /* ------------- upload ------------------ */
  const handleSubmit = async (e) => {
    e.preventDefault();
    // upload step: reset upload alerts, keep save alerts intact
    setUploadError(null);
    setUploadSuccess(null);

    if (!file) {
      setUploadError("Please select a file.");
      return;
    }

    const problem = validateFile(file);
    if (problem) {
      setUploadError(problem);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true);
    try {
      const res = await fetch("/api/upload-csv", {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      if (!res.ok) {
        if (res.status === 401) throw new Error("Please log in again.");
        if (res.status === 400) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Invalid CSV file.");
        }
        throw new Error(`Upload failed (${res.status}).`);
      }

      const data = await res.json();
      setInsights(data);
      setS3Key(data.s3_key ?? null);
      setUploadSuccess("Upload processed. Preview generated.");
      // do not scroll here; the user is already near the top form
    } catch (err) {
      console.error(err);
      setUploadError(err.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  /* ------------- save dataset ------------ */
  const handleSave = async (e) => {
    e.preventDefault();
    // save step: only clear save alerts; don't wipe upload alerts
    clearSaveAlerts();

    if (!canSave) return;

    const payload = {
      title: title.trim(),
      description: description.trim(),
      filename: file.name,
      s3_key: s3Key,
      // metadata from insights (defensive guards)
      n_rows: insights?.n_rows ?? null,
      n_columns: insights?.n_columns ?? null,
      has_missing_values: insights?.has_missing_values ?? null,
      column_metadata: insights?.column_metadata ?? null,
      current_stage: "uploaded",
    };

    setIsSaving(true);
    try {
      const res = await fetch("/api/datasets/save", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        if (res.status === 401)
          throw new Error("Not authenticated – please log in again.");
        const errPayload = await res.json().catch(() => ({}));
        throw new Error(errPayload.detail ?? `Save failed (${res.status}).`);
      }

      const data = await res.json();
      setSaveSuccess(
        `Dataset saved! ID: ${data.id} Go to Datasets to find it!`
      );
      pingBottomAlert(); // scroll to the alert under the Save button
    } catch (err) {
      console.error("Save error:", err);
      setSaveError(err.message || "Save failed.");
      pingBottomAlert(); // scroll to the alert under the Save button
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setTitle("");
    setDescription("");
    setFile(null);
    setInsights(null);
    setS3Key(null);
    clearAllAlerts();
  };

  /* ------------- UI helpers ---------------------- */
  const previewRows = insights?.preview ?? [];
  const previewCols =
    previewRows.length > 0
      ? Object.keys(previewRows[0])
      : insights?.columns ?? [];
  const nullCounts = insights?.null_counts ?? {};
  const infoOutput = insights?.info_output ?? "";
  const shapeRows = insights?.shape?.[0] ?? insights?.n_rows ?? null;
  const shapeCols = insights?.shape?.[1] ?? insights?.n_columns ?? null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="rounded-2xl bg-white p-6 sm:p-8 shadow-sm ring-1 ring-black/5">
        <h2 className="text-2xl font-semibold tracking-tight text-gray-900">
          Upload a CSV
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Add a title and short description, then drop in your .csv to preview
          it before saving.
        </p>

        {/* Upload alerts (top) */}
        <div className="mt-4 space-y-2" aria-live="polite">
          {uploadError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700 text-sm">
              {uploadError}
            </div>
          )}
          {uploadSuccess && (
            <div className="rounded-md border border-emerald-200 bg-emerald-50 px-4 py-2 text-emerald-700 text-sm">
              {uploadSuccess}
            </div>
          )}
        </div>

        {/* Upload form */}
        <form onSubmit={handleSubmit} className="mt-6 grid gap-6">
          {/* Title / Description */}
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Title <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => {
                  clearAllAlerts();
                  setTitle(e.target.value);
                }}
                className="mt-1 block w-full rounded-lg border border-gray-300 bg-white p-2.5 text-gray-900 placeholder:text-gray-400 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                placeholder="e.g., Customer Churn 2024 Q3"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Description <span className="text-red-500">*</span>
              </label>
              <textarea
                value={description}
                onChange={(e) => {
                  clearAllAlerts();
                  setDescription(e.target.value);
                }}
                rows={3}
                className="mt-1 block w-full rounded-lg border border-gray-300 bg-white p-2.5 text-gray-900 placeholder:text-gray-400 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                placeholder="Short summary of what this dataset contains."
                required
              />
            </div>
          </div>

          {/* Dropzone + picker */}
          <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={[
              "relative rounded-2xl border-2 border-dashed p-6 transition",
              isDragging
                ? "border-indigo-500 bg-indigo-50/50"
                : "border-gray-300 bg-gray-50",
            ].join(" ")}
          >
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-gray-200">
                {/* upload icon */}
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 text-gray-700"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path d="M3 15v3a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-3" />
                  <path d="M12 3v12" />
                  <path d="m7 8 5-5 5 5" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-700">
                  Drag & drop your CSV here, or{" "}
                  <label className="font-medium text-indigo-600 hover:underline cursor-pointer">
                    browse
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={handleInputChange}
                      className="sr-only"
                    />
                  </label>
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Only .csv files up to {MAX_FILE_SIZE_MB}MB.
                </p>
              </div>
              {file && (
                <div className="rounded-lg bg-white px-3 py-2 text-xs text-gray-700 ring-1 ring-gray-200 shadow-sm">
                  <div
                    className="truncate max-w-[220px]"
                    title={fileMeta?.name}
                  >
                    {fileMeta?.name}
                  </div>
                  <div className="text-gray-500">
                    {fileMeta?.sizeMB} MB &middot; {fileMeta?.type}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Upload actions */}
          <div className="flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-end">
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              disabled={isUploading || isSaving}
            >
              Reset
            </button>

            <button
              type="submit"
              disabled={!canUpload}
              className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50"
            >
              {isUploading ? (
                <>
                  <Spinner className="mr-2" /> Uploading…
                </>
              ) : (
                "Upload & Preview"
              )}
            </button>
          </div>
        </form>

        {/* Insights / Preview */}
        {insights && (
          <div className="mt-10 space-y-8">
            {/* Summary cards */}
            <div className="grid gap-4 sm:grid-cols-3">
              <StatCard label="Rows" value={shapeRows ?? "—"} />
              <StatCard label="Columns" value={shapeCols ?? "—"} />
              <StatCard
                label="Missing Values"
                value={insights?.has_missing_values ? "Yes" : "No"}
              />
            </div>

            {/* Columns list */}
            {Array.isArray(insights?.columns) &&
              insights.columns.length > 0 && (
                <div>
                  <h3 className="text-base font-semibold text-gray-800">
                    Columns
                  </h3>
                  <p className="mt-1 text-sm text-gray-600 break-words">
                    {insights.columns.join(", ")}
                  </p>
                </div>
              )}

            {/* Preview table */}
            <div>
              <h3 className="text-base font-semibold text-gray-800 mb-2">
                Preview Rows
              </h3>
              <div className="overflow-auto max-h-80 rounded-lg ring-1 ring-gray-200">
                <table className="min-w-full text-xs">
                  <thead className="bg-gray-50 sticky top-0 z-10">
                    <tr>
                      {previewCols.map((col) => (
                        <th
                          key={col}
                          className="px-3 py-2 text-left font-medium text-gray-700 border-b"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.length === 0 ? (
                      <tr>
                        <td
                          className="px-3 py-3 text-gray-500 text-center"
                          colSpan={previewCols.length || 1}
                        >
                          No preview rows available.
                        </td>
                      </tr>
                    ) : (
                      previewRows.map((row, i) => (
                        <tr key={i} className="odd:bg-white even:bg-gray-50">
                          {previewCols.map((col, j) => (
                            <td
                              key={`${i}-${j}`}
                              className="px-3 py-2 border-b text-gray-700"
                            >
                              {formatCell(row[col])}
                            </td>
                          ))}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Missing values table */}
            {Object.keys(nullCounts).length > 0 && (
              <div className="bg-white rounded-xl ring-1 ring-gray-200 shadow-sm">
                <div className="px-4 py-3 border-b">
                  <h3 className="text-base font-semibold text-gray-800">
                    Missing Values
                  </h3>
                </div>
                <div className="overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left font-medium text-gray-700">
                          Column
                        </th>
                        <th className="px-4 py-2 text-right font-medium text-gray-700">
                          Missing Count
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {Object.entries(nullCounts).map(([col, count]) => (
                        <tr key={col} className="hover:bg-gray-50">
                          <td className="px-4 py-2 font-medium text-gray-800">
                            {col}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-600">
                            {count}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* df.info() */}
            {infoOutput && (
              <div>
                <h3 className="text-base font-semibold text-gray-800">
                  df.info()
                </h3>
                <pre className="mt-2 overflow-auto max-h-64 bg-gray-50 p-4 rounded-lg text-xs text-gray-800 whitespace-pre-wrap ring-1 ring-gray-200">
                  {infoOutput}
                </pre>
              </div>
            )}

            {/* Save area (with bottom alerts) */}
            <form onSubmit={handleSave} className="pt-4 border-t">
              <div className="flex items-center justify-end gap-3">
                <span className="text-xs text-gray-500">
                  {s3Key ? "File stored." : "File not stored yet."}
                </span>
                <button
                  type="submit"
                  disabled={!canSave}
                  className="inline-flex items-center justify-center rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-500 disabled:opacity-50"
                >
                  {isSaving ? (
                    <>
                      <Spinner className="mr-2" /> Saving…
                    </>
                  ) : (
                    "Save to Database"
                  )}
                </button>
              </div>

              {/* Bottom alerts pinned below Save */}
              <div
                ref={bottomAlertRef}
                className="mt-3 space-y-2"
                aria-live="polite"
                role="status"
              >
                {saveError && (
                  <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700 text-sm">
                    {saveError}
                  </div>
                )}
                {saveSuccess && (
                  <div className="rounded-md border border-emerald-200 bg-emerald-50 px-4 py-2 text-emerald-700 text-sm">
                    {saveSuccess}
                  </div>
                )}
              </div>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- small UI bits ---------------- */
function StatCard({ label, value }) {
  return (
    <div className="rounded-xl bg-gray-50 p-4 ring-1 ring-gray-200">
      <div className="text-xs uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className="mt-1 text-lg font-semibold text-gray-900">
        {value ?? "—"}
      </div>
    </div>
  );
}

function Spinner({ className = "" }) {
  return (
    <svg
      className={`h-4 w-4 animate-spin ${className}`}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="4"
      />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function formatCell(val) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") return Number.isFinite(val) ? String(val) : "—";
  const s = String(val);
  return s.length > 160 ? s.slice(0, 157) + "…" : s;
}
