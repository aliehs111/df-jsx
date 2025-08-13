// src/pages/Dashboard.jsx
import { Link } from "react-router-dom";
import KPICards from "../components/KPICards";
import RecentDatasets from "../components/RecentDatasets";
import QuickActions from "../components/QuickActions";
import newlogo500 from "../assets/newlogo500.png";

const logo = newlogo500;

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-neutralLight">
      {/* Header */}
      <header className="relative overflow-hidden bg-gradient-to-r from-primary via-primary/90 to-secondary py-10 px-8 sm:px-20 shadow-md">
        <div className="relative flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          {/* Left: Title + subtitle + version pill */}
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white drop-shadow-sm">
                Dashboard
              </h1>
              <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                v0.9 • dev
              </span>
            </div>
            <p className="mt-2 text-cyan-100">
              Manage your datasets and run quick models
            </p>

            {/* Optional mini-breadcrumbs */}
            <nav className="mt-3 text-xs text-white/70">
              <ol className="flex items-center gap-2">
                <li>Home</li>
                <li className="opacity-60">/</li>
                <li className="font-medium text-white">Dashboard</li>
              </ol>
            </nav>
          </div>

          {/* Right: Logo + primary actions */}
          <div className="flex items-center gap-4">
            {/* Actions (show first on small screens) */}
            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 order-2 sm:order-1">
              <Link
                to="/upload"
                className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-4 py-2 text-white text-sm font-semibold shadow-sm hover:bg-accent/90"
              >
                {/* upload icon */}
                <svg
                  viewBox="0 0 24 24"
                  className="h-5 w-5"
                  fill="currentColor"
                >
                  <path d="M12 3a1 1 0 011 1v8h3l-4 4-4-4h3V4a1 1 0 011-1zM5 19h14a1 1 0 100-2H5a1 1 0 000 2z" />
                </svg>
                Upload CSV
              </Link>
              <Link
                to="/datasets"
                className="inline-flex items-center justify-center gap-2 rounded-md border border-white/30 px-4 py-2 text-white/90 text-sm font-semibold hover:bg-white/10"
              >
                View Datasets
              </Link>
            </div>

            {/* Logo */}
            <img
              src={logo}
              alt="Logo"
              className="order-1 sm:order-2 h-32 w-32 sm:h-36 sm:w-36 rounded-xl object-contain bg-white/10 ring-1 ring-white/30 shadow-lg"
            />
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8 space-y-8">
        {/* KPIs */}
        <section>
          <KPICards />
        </section>

        {/* Two-column: Recent datasets + Quick actions */}
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl bg-white p-6 shadow-sm ring-1 ring-black/5 transition hover:shadow-md hover:ring-black/10">
            <RecentDatasets />
          </div>
          <div className="rounded-xl bg-white p-6 shadow-sm ring-1 ring-black/5 transition hover:shadow-md hover:ring-black/10">
            <QuickActions />
          </div>
        </section>

        {/* Explainer video */}
        <section className="rounded-xl bg-white p-6 shadow-sm ring-1 ring-black/5 flex items-center justify-between">
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
            className="flex items-center space-x-2 rounded-md border px-4 py-2 text-sm bg-primary text-white opacity-70 cursor-not-allowed"
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
        </section>
      </main>
    </div>
  );
}
