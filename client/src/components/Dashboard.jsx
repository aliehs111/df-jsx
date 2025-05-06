// src/pages/Dashboard.jsx
import Navbar from '../components/Navbar'
import QuickStart   from '../components/QuickStart'
import KPICards     from '../components/KPICards'
import RecentDatasets from '../components/RecentDatasets'
import QuickActions from '../components/QuickActions'

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* <Navbar /> */}

      <header className="bg-cyan-300 py-8 pb-32">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-bold text-white">Dashboard</h1>
        </div>
      </header>

      <main className="-mt-12 mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8 space-y-6">
        {/* <QuickStart /> */}
        <KPICards />

        <div className="grid gap-6 lg:grid-cols-2">
          <RecentDatasets />
          <QuickActions />
        </div>

        {/* Placeholder for future explainer video */}
        <div className="rounded-lg bg-white p-6 shadow flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">🎬 App Explainer</h3>
            <p className="mt-1 text-sm text-gray-600">
              A quick 2-minute video is coming soon!
            </p>
          </div>
          <button
            disabled
            className="flex items-center space-x-2 rounded-md border px-4 py-2 text-sm text-gray-400"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none" viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-4.36-2.507A1 1 0 009 9.6v4.8a1 1 0 001.392.92l4.36-2.507a1 1 0 000-1.732z" />
            </svg>
            <span>Play</span>
          </button>
        </div>
      </main>
    </div>
  )
}

