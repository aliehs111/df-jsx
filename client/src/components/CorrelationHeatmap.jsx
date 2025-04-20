import React, { useState, useEffect } from "react";

const CorrelationHeatmap = ({ datasetId }) => {
  const [heatmap, setHeatmap] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`http://localhost:8000/datasets/${datasetId}/correlation`)
      .then(res => res.json())
      .then(data => setHeatmap(data.heatmap))
      .catch(err => {
        console.error(err);
        setError("Failed to load heatmap");
      });
  }, [datasetId]);

  if (error) return <p className="text-red-600">{error}</p>;

  return heatmap ? (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-gray-800">Correlation Heatmap</h3>
      <img src={heatmap} alt="Correlation Heatmap" className="mt-4 border rounded shadow" />
    </div>
  ) : (
    <p className="text-gray-500">Loading heatmap...</p>
  );
};

export default CorrelationHeatmap;
