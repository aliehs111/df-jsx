import { useEffect, useState } from 'react'
import { CalendarIcon, EyeIcon } from '@heroicons/react/20/solid'
import { Link } from 'react-router-dom'

export default function DatasetsList() {
  const [datasets, setDatasets] = useState([])

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const res = await fetch('http://localhost:8000/datasets')
        const data = await res.json()
        setDatasets(data)
      } catch (err) {
        console.error('Failed to fetch datasets:', err)
      }
    }

    fetchDatasets()
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">My Datasets</h2>
      <ul role="list" className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {datasets.map((dataset) => (
          <li
            key={dataset.id}
            className="col-span-1 divide-y divide-gray-200 rounded-lg bg-white shadow"
          >
            <div className="flex flex-col space-y-3 p-6">
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 truncate">{dataset.title}</h3>
                <p className="mt-1 text-sm text-gray-600 truncate">{dataset.description}</p>
              </div>
              <div className="flex items-center text-sm text-gray-500">
                <CalendarIcon className="h-5 w-5 mr-1" aria-hidden="true" />
                {new Date(dataset.uploaded_at).toLocaleDateString()}
              </div>
            </div>
            <div className="-mt-px flex divide-x divide-gray-200">
              <div className="flex w-0 flex-1">
                <Link
                  to={`/datasets/${dataset.id}`}
                  className="relative -mr-px inline-flex w-0 flex-1 items-center justify-center gap-x-3 rounded-bl-lg border border-transparent py-4 text-sm font-semibold text-indigo-600 hover:text-indigo-900"
                >
                   {dataset.title}
                  <EyeIcon className="h-5 w-5 text-indigo-400" aria-hidden="true" />
                  View
                </Link>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
