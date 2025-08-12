// src/components/KPICards.jsx
import { useEffect, useState } from "react";

export default function KPICards() {
  const [kpis, setKpis] = useState({
    totalDatasets: 0,
    uploadsToday: 0,
    uploadsWeek: 0,
    lastUpload: "—",
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ac = new AbortController();

    const parseUploadTime = (s3key) => {
      if (typeof s3key !== "string") return null;
      // Expect: uploads/20250808T015750_filename.csv
      const m = s3key.match(/uploads\/(\d{8}T\d{6})/);
      if (!m) return null;
      const ts = m[1]; // YYYYMMDDTHHMMSS
      const yyyy = Number(ts.slice(0, 4));
      const mm = Number(ts.slice(4, 6));
      const dd = Number(ts.slice(6, 8));
      const HH = Number(ts.slice(9, 11));
      const MM = Number(ts.slice(11, 13));
      const SS = Number(ts.slice(13, 15));
      // Local time is fine for dashboard rollups
      return new Date(yyyy, mm - 1, dd, HH, MM, SS);
    };

    async function load() {
      try {
        setLoading(true);
        const res = await fetch("/api/datasets", {
          credentials: "include",
          signal: ac.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const list = (await res.json()) ?? [];
        const totalDatasets = Array.isArray(list) ? list.length : 0;

        const times = (Array.isArray(list) ? list : [])
          .map((d) => parseUploadTime(d?.s3_key))
          .filter(Boolean);

        const now = new Date();
        const startOfToday = new Date(
          now.getFullYear(),
          now.getMonth(),
          now.getDate()
        );
        const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

        const uploadsToday = times.filter((t) => t >= startOfToday).length;
        const uploadsWeek = times.filter((t) => t >= oneWeekAgo).length;

        const latest =
          times.length > 0
            ? times.reduce((max, t) => (t > max ? t : max), times[0])
            : null;

        const lastUpload =
          latest != null
            ? new Intl.DateTimeFormat("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              }).format(latest)
            : "—";

        setKpis({ totalDatasets, uploadsToday, uploadsWeek, lastUpload });
      } catch (err) {
        if (err.name !== "AbortError") {
          console.warn("KPIs load failed:", err);
          setKpis((k) => ({ ...k, lastUpload: "—" }));
        }
      } finally {
        setLoading(false);
      }
    }

    load();
    return () => ac.abort();
  }, []);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-blue-800">
      {loading ? (
        <>
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </>
      ) : (
        <>
          <Card title="📦 Total Datasets" value={kpis.totalDatasets} />
          <Card title="📈 Uploads Today" value={kpis.uploadsToday} />
          <Card title="📅 Uploads This Week" value={kpis.uploadsWeek} />
          <Card title="⏱️ Last Upload" value={kpis.lastUpload} />
        </>
      )}
    </div>
  );
}

function Card({ title, value }) {
  const display =
    typeof value === "number" ? value.toLocaleString() : value || "—";

  return (
    <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
      <h3 className="text-sm font-medium text-blue-800">{title}</h3>
      <p className="mt-2 text-2xl font-semibold text-blue-900">{display}</p>
      <p className="mt-1 text-xs text-gray-500">Live from uploads</p>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-lg bg-white p-6 shadow border border-gray-100 animate-pulse">
      <div className="h-3 w-24 bg-gray-200 rounded"></div>
      <div className="mt-3 h-7 w-20 bg-gray-200 rounded"></div>
      <div className="mt-2 h-3 w-28 bg-gray-100 rounded"></div>
    </div>
  );
}
