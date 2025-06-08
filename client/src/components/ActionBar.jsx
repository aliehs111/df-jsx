// src/components/ActionBar.jsx
import React from "react";
import { Link } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/24/outline";

export default function ActionBar({
  id,
  hasClean,
  view,
  onViewChange,
  onFetchInsights,
  onFetchHeatmap,
  onDownloadCleaned,
}) {
  const btnBase = "px-4 py-2 rounded";
  const active = "bg-blue-600 text-white";
  const inactive = "bg-gray-200 text-gray-600";

  return (
    <div className="mb-6 flex flex-wrap justify-center gap-4">
      <button
        onClick={() => onViewChange("raw")}
        className={`${btnBase} ${view === "raw" ? active : inactive}`}
      >
        Raw
      </button>

      {hasClean && (
        <button
          onClick={() => onViewChange("cleaned")}
          className={`${btnBase} ${view === "cleaned" ? active : inactive}`}
        >
          Cleaned
        </button>
      )}

      <button
        onClick={() => {
          onFetchInsights();
          onViewChange("insights");
        }}
        className={`${btnBase} ${view === "insights" ? active : inactive}`}
      >
        Insights
      </button>

      <button
        onClick={() => {
          onFetchHeatmap();
          onViewChange("heatmap");
        }}
        className={`${btnBase} ${view === "heatmap" ? active : inactive}`}
      >
        Heatmap
      </button>

      <Link to={`/datasets/${id}/clean`} className={`${btnBase} ${inactive}`}>
        Pipeline Sandbox
      </Link>

      {hasClean && (
        <button
          onClick={onDownloadCleaned}
          className={`${btnBase} ${inactive}`}
        >
          Download Cleaned CSV
        </button>
      )}

      <Link
        to="/chat"
        className={`${btnBase} ${inactive} inline-flex items-center space-x-1`}
      >
        <ChatBubbleLeftEllipsisIcon className="h-4 w-4" />
        <span>Chat</span>
        <img src={newlogo500} alt="Data Tutor" className="h-4 w-4" />
      </Link>
    </div>
  );
}
