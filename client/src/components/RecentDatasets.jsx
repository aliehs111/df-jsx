// src/components/RecentDatasets.jsx
import { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";

export default function RecentDatasets() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  // Parse time from s3_key like: uploads/20250808T015750_filename.csv
  const parseUploadTime = (s3key, fallback) => {
    if (typeof s3key === "string") {
      const m = s3key.match(/uploads\/(\d{8}T\d{6})/);
      if (m) {
        const ts = m[1]; // YYYYMMDDTHHMMSS
        const yyyy = Number(ts.slice(0, 4));
        const mm = Number(ts.slice(4, 6));
        const dd = Number(ts.slice(6, 8));
        const HH = Number(ts.slice(9, 11));
        const MM = Number(ts.slice(11, 13));
        const SS = Number(ts.slice(13, 15));
        return new Date(yyyy, mm - 1, dd, HH, MM, SS);
      }
    }
    // Fallback to created_at/updated_at if present
    const dt =
      (fallback?.updated_at && new Date(fallback.updated_at)) ||
      (fallback?.created_at && new Date(fallback.created_at)) ||
      null;
    return Number.isNaN(dt?.getTime()) ? null : dt;
  };

  const timeAgo = (date) => {
    if (!date) return "—";
    const now = new Date();
    const diffMs = now - date;
    const s = Math.floor(diffMs / 1000);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    const d = Math.floor(h / 24);
    if (d > 7) {
      // show short date if older than a week
      return new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
      }).format(date);
    }
    if (d >= 1) return `${d}d ago`;
    if (h >= 1) return `${h}h ago`;
    if (m >= 1) return `${m}m ago`;
    return "just now";
  };

  useEffect(() => {
    const ac = new AbortController();
    async function load() {
      try {
        setLoading(true);
        const res = await fetch("/api/datasets", {
          credentials: "include",
          signal: ac.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const list = (await res.json()) ?? [];
        setItems(Array.isArray(list) ? list : []);
      } catch (err) {
        if (err.name !== "AbortError") {
          console.warn("RecentDatasets load failed:", err);
          setItems([]);
        }
      } finally {
        setLoading(false);
      }
    }
    load();
    return () => ac.abort();
  }, []);

  const recent5 = useMemo(() => {
    const enriched = items.map((d) => {
      const uploadedAt = parseUploadTime(d?.s3_key, d);
      return { ...d, uploadedAt };
    });
    enriched.sort((a, b) => {
      const at = a.uploadedAt ? a.uploadedAt.getTime() : 0;
      const bt = b.uploadedAt ? b.uploadedAt.getTime() : 0;
      return bt - at;
    });
    return enriched.slice(0, 3);
  }, [items]);

  if (loading) {
    return (
      <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-blue-800">
            Recent datasets
          </h3>
          <div className="h-6 w-20 bg-gray-200 rounded animate-pulse" />
        </div>
        <ul className="mt-4 space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <li
              key={i}
              className="flex items-center justify-between p-3 rounded border border-gray-100 animate-pulse"
            >
              <div className="space-y-2">
                <div className="h-4 w-44 bg-gray-200 rounded" />
                <div className="h-3 w-24 bg-gray-100 rounded" />
              </div>
              <div className="h-8 w-28 bg-gray-200 rounded" />
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (!recent5.length) {
    return (
      <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
        <h3 className="text-lg font-semibold text-blue-800">Recent datasets</h3>
        <div className="mt-4 rounded border border-dashed p-6 text-center text-gray-600">
          No datasets yet.{" "}
          <Link to="/upload" className="text-blue-700 underline">
            Upload a CSV
          </Link>{" "}
          to get started.
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-blue-800">Recent datasets</h3>
        <Link to="/datasets" className="text-sm text-blue-700 hover:underline">
          View all
        </Link>
      </div>

      <ul className="mt-4 space-y-3">
        {recent5.map((d) => (
          <li
            key={d.id}
            className="flex items-center justify-between p-3 rounded border border-gray-100 hover:border-blue-200 transition"
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate font-medium text-blue-900">
                  {d.title || d.filename || `Dataset #${d.id}`}
                </span>
                {/* Optional chips */}
                {d.has_missing_values === true && <Chip>Has missing</Chip>}
                {d.target_column && <Chip>Target set</Chip>}
                {d.current_stage && <Chip>{d.current_stage}</Chip>}
              </div>
              <div className="mt-1 text-xs text-gray-500">
                {d.filename ? d.filename : "—"} • Updated{" "}
                {timeAgo(d.uploadedAt)}
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <Link
                to={`/datasets/${d.id}`}
                className="px-3 py-1.5 text-sm rounded border border-gray-200 hover:bg-gray-50"
              >
                View
              </Link>
              <Link
                to={`/datasets/${d.id}/clean`}
                className="px-3 py-1.5 text-sm rounded border border-gray-200 hover:bg-gray-50"
              >
                Clean
              </Link>
              <Link
                to={`/models?dataset=${d.id}`}
                className="px-3 py-1.5 text-sm rounded border border-gray-200 hover:bg-gray-50"
              >
                Run model
              </Link>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Chip({ children }) {
  return (
    <span className="inline-flex items-center rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700">
      {children}
    </span>
  );
}
