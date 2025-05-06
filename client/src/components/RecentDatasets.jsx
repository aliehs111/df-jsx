// src/components/RecentDatasets.jsx
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

export default function RecentDatasets() {
  const [datasets, setDatasets] = useState([])

  useEffect(() => {
    // TODO: replace with real fetch
    // fetch('/api/datasets?limit=5').then(r => r.json()).then(setDatasets)
    setDatasets([
      { id: 1, name: 'sales_q1.csv', rows: 1200, cols: 12, uploaded: 'May 5, 2025' },
      { id: 2, name: 'expenses.csv',  rows: 800, cols:  8, uploaded: 'May 4, 2025' },
      // …
    ])
  }, [])

  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <h2 className="text-xl font-semibold mb-4">Recent Datasets</h2>
      <ul className="space-y-3">
        {datasets.map(ds => (
          <li key={ds.id} className="flex justify-between items-center">
            <Link
              to={`/datasets/${ds.id}`}
              className="text-cyan-700 hover:underline"
            >
              {ds.name}
            </Link>
            <span className="text-sm text-gray-500">
              {ds.rows}×{ds.cols} • {ds.uploaded}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
