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
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card title="📦 Total Datasets" value={kpis.totalDatasets} />
      <Card title="🔢 Rows Processed"  value={kpis.totalRows.toLocaleString()} />
      <Card title="📊 Total Columns"  value={kpis.totalCols} />
      <Card title="⏱️ Last Upload"     value={kpis.lastUpload} />
    </div>
  )
}

function Card({ title, value }) {
  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className="mt-2 text-2xl font-semibold text-gray-900">{value}</p>
    </div>
  )
}
