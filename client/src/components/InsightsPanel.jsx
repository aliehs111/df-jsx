// client/src/components/InsightsPanel.jsx
import React from "react";

export default function InsightsPanel({ insights }) {
  return (
    <div className="mt-6 bg-gray-50 p-4 rounded-md">
      {/* Dataset Summary */}
      <h3 className="text-lg font-semibold text-gray-700 mb-2">
        Dataset Summary
      </h3>
      <p className="mb-1">
        <strong>Shape:</strong> {insights.shape[0]} × {insights.shape[1]}
      </p>
      <p className="mb-4">
        <strong>Columns:</strong> {insights.columns.join(", ")}
      </p>

      {/* df.info() Output */}
      <pre className="overflow-auto max-h-96 bg-gray-100 p-4 rounded text-xs whitespace-pre-wrap">
        {insights.info_output}
      </pre>
    </div>
  );
}
