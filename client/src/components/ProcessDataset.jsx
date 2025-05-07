import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'

export default function ProcessDataset() {
  const { id } = useParams()
  const navigate = useNavigate()

  const [clean, setClean]           = useState({
    dropna: false,
    fillna: {},
    lowercase_headers: false,
    remove_duplicates: false,
  })
  const [preprocess, setPreprocess] = useState({ scale: '', encoding: '' })
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)

  const handleRun = async () => {
    const token = localStorage.getItem('token')
    const res = await fetch(`/datasets/${id}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ clean, preprocess }),
    })

    if (!res.ok) {
      setError('Processing failed')
      return
    }
    const data = await res.json()
    setResult(data)
  }

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow rounded space-y-6">
      <h1 className="text-2xl font-bold">Clean &amp; Preprocess Dataset</h1>

      Cleaning controls
      <section className="space-y-2">
        <h2 className="font-semibold">1. Cleaning</h2>
        <label className="inline-flex items-center">
          <input
            type="checkbox"
            checked={clean.dropna}
            onChange={(e) => setClean(c => ({ ...c, dropna: e.target.checked }))}
            className="mr-2"
          />
          Drop rows with any nulls
        </label>

        <label className="inline-flex items-center">
          <input
            type="checkbox"
            checked={clean.lowercase_headers}
            onChange={(e) => setClean(c => ({ ...c, lowercase_headers: e.target.checked }))}
            className="mr-2"
          />
          Lowercase column names
        </label>

        <label className="inline-flex items-center">
          <input
            type="checkbox"
            checked={clean.remove_duplicates}
            onChange={(e) => setClean(c => ({ ...c, remove_duplicates: e.target.checked }))}
            className="mr-2"
          />
          Remove duplicate rows
        </label>

        Fill‐nulls: simple text input for JSON
        <div>
          <label className="block text-sm font-medium">Fill missing values (JSON)</label>
          <input
            type="text"
            placeholder='e.g. {"age": 0, "salary": 100}'
            onBlur={(e) => {
              try {
                const obj = JSON.parse(e.target.value)
                setClean(c => ({ ...c, fillna: obj }))
              } catch {
                alert('Invalid JSON')
              }
            }}
            className="mt-1 w-full border px-2 py-1 rounded"
          />
        </div>
      </section>

      Preprocessing controls
      <section className="space-y-2">
        <h2 className="font-semibold">2. Preprocessing</h2>

        <div>
          <label className="block text-sm font-medium">Scaling</label>
          <select
            value={preprocess.scale}
            onChange={e => setPreprocess(p => ({ ...p, scale: e.target.value }))}
            className="mt-1 w-full border px-2 py-1 rounded"
          >
            <option value="">None</option>
            <option value="normalize">Min–Max Normalize</option>
            <option value="standardize">Z-score Standardize</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium">Encoding</label>
          <select
            value={preprocess.encoding}
            onChange={e => setPreprocess(p => ({ ...p, encoding: e.target.value }))}
            className="mt-1 w-full border px-2 py-1 rounded"
          >
            <option value="">None</option>
            <option value="onehot">One-Hot</option>
            <option value="label">Label</option>
          </select>
        </div>
      </section>

      <button
        onClick={handleRun}
        className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-500"
      >
        Run &amp; Save Final CSV
      </button>

      {error && <p className="text-red-600">{error}</p>}

      {result && (
        <div className="p-4 bg-green-100 rounded">
          <p><strong>Done!</strong> Final CSV in S3 at <code>{result.s3_key}</code></p>
        </div>
      )}
    </div>
  )
}
