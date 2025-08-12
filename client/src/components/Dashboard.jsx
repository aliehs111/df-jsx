// src/pages/Dashboard.jsx
import KPICards from "../components/KPICards";
import RecentDatasets from "../components/RecentDatasets";
import QuickActions from "../components/QuickActions";
import newlogo500 from "../assets/newlogo500.png";

const logo = newlogo500;

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-cyan-200 py-10 flex items-center justify-between px-8 sm:px-20 border-b border-cyan-300/40">
        <div>
          <h1 className="text-5xl font-bold tracking-tight text-white">
            Dashboard
          </h1>
          <p className="text-blue-900 mt-2">
            Manage your datasets and run quick models
          </p>
        </div>
        <img
          src={logo}
          alt="Logo"
          className="h-40 w-40 rounded-md object-contain bg-white/10 ring-1 ring-white/30"
        />
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8 space-y-8 bg-cyan-50">
        {/* KPIs */}
        <KPICards />

        {/* Two-column: Recent datasets + Quick actions */}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
            <RecentDatasets />
          </div>
          <div className="rounded-lg bg-white p-6 shadow border border-gray-100">
            <QuickActions />
          </div>
        </div>

        {/* Explainer video */}
        <div className="rounded-lg bg-white p-6 shadow border border-gray-100 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold">🎬 Explainer Video</h3>
              <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700 ring-1 ring-inset ring-blue-200">
                Coming soon
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-600">
              A quick 2-minute video is coming soon!
            </p>
          </div>
          <button
            disabled
            className="flex items-center space-x-2 rounded-md border px-4 py-2 text-sm bg-blue-800 text-white opacity-70 cursor-not-allowed"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14.752 11.168l-4.36-2.507A1 1 0 009 9.6v4.8a1 1 0 001.392.92l4.36-2.507a1 1 0 000-1.732z"
              />
            </svg>
            <span>Play</span>
          </button>
        </div>
      </main>
    </div>
  );
}
