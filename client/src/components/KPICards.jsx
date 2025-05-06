// src/components/KPICards.jsx
import { useEffect, useState } from 'react'

export default function KPICards() {
  const [kpis, setKpis] = useState({
    totalDatasets: 0,
    totalRows: 0,
    totalCols: 0,
    lastUpload: '',
  })

  useEffect(() => {
    // TODO: fetch these values from your backend (MySQL/S3)
    // Example:
    // fetch('/api/kpis').then(r => r.json()).then(setKpis)
    setKpis({
      totalDatasets: 12,
      totalRows: 1345678,
      totalCols: 53,
      lastUpload: 'May 6, 2025',
    })
  }, [])

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-blue-800">
      <Card title="📦 Total Datasets" value={kpis.totalDatasets} />
      <Card title="🔢 Rows Processed"  value={kpis.totalRows.toLocaleString()} />
      <Card title="📊 Total Columns"  value={kpis.totalCols} />
      <Card title="⏱️ Last Upload"     value={kpis.lastUpload} />
    </div>
  )
}

function Card({ title, value }) {
  return (
    <div className="rounded-lg bg-white p-6 shadow text-blue-800">
      <h3 className="text-sm font-medium text-blue-800">{title}</h3>
      <p className="mt-2 text-2xl font-semibold ext-blue-800">{value}</p>
    </div>
  )
}
