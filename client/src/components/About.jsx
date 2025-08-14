import { Link } from "react-router-dom";
import logo from "../assets/newlogo500.png"; // Adjust path to match your setup
import ArchitectureDiagram from "./ArchitectureDiagram";

export default function About() {
  return (
    <div className="min-h-screen bg-neutralLight">
      {/* Header */}
      <header className="relative overflow-hidden bg-gradient-to-r from-primary via-primary/90 to-secondary py-10 px-8 sm:px-20 shadow-md">
        <div className="relative flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white drop-shadow-sm">
                About df.jsx
              </h1>
              <span className="rounded-full bg-white/15 px-2 py-0.5 text-[11px] font-medium text-white/90 ring-1 ring-white/25">
                v0.9 • dev
              </span>
            </div>
            <p className="mt-2 text-cyan-100 text-sm">
              The story behind df.jsx and its mission
            </p>
          </div>
          <img
            src={logo}
            alt="df.jsx Logo"
            className="h-32 w-32 sm:h-36 sm:w-36 rounded-xl object-contain bg-white/10 ring-1 ring-white/30 shadow-lg"
          />
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="rounded-xl bg-white p-8 shadow-sm ring-1 ring-black/5 transition hover:shadow-md hover:ring-black/10">
          <p className="text-gray-600 text-sm mb-6 leading-relaxed">
            <strong>df.jsx</strong>—short for "dataframes in React"—is a
            full-stack academic project built to explore modern data science
            workflows in a web environment. It demonstrates how a Python-based
            analytics backend can integrate seamlessly with a React frontend for
            real-time model inference and data visualization.
          </p>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-primary mb-3">Purpose</h2>
            <p className="text-gray-600 text-sm leading-relaxed">
              The goal was to step outside a JavaScript-only comfort zone and
              build a production-style application using a Python{" "}
              <strong>FastAPI</strong>
              backend. This backend handles data ingestion, preprocessing, and
              machine learning model execution using <strong>pandas</strong>,
              <strong> NumPy</strong>, <strong>scikit-learn</strong>, and
              <strong> Matplotlib</strong>. The project serves as a
              proof-of-concept for delivering statistical summaries,
              visualizations, and predictive analytics entirely through API
              calls.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-primary mb-3">
              Architecture
            </h2>
            <p className="text-gray-600 text-sm leading-relaxed">
              The backend is built with FastAPI, using async SQLAlchemy to
              connect to a MySQL database (JawsDB on Heroku) for storing dataset
              metadata and model parameters. Uploaded CSVs are temporarily
              stored in memory for processing, with finalized datasets persisted
              to <strong>AWS S3</strong>. The frontend is a{" "}
              <strong>React</strong> application styled with
              <strong> TailwindCSS</strong> and <strong>Headless UI</strong>,
              consuming the backend APIs via <strong>Axios</strong>. Model
              results and plots are rendered dynamically, and interactive
              components are used to trigger analysis without page reloads.
              <ArchitectureDiagram />
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-xl font-semibold text-primary mb-3">
              Key Features
            </h2>
            <ul className="list-disc list-inside text-gray-600 text-sm leading-relaxed space-y-2">
              <li>
                CSV upload with immediate preview and dataframe shape/info
                summary
              </li>
              <li>
                Data cleaning options: missing value handling, normalization,
                categorical encoding
              </li>
              <li>
                Real-time predictive models, including regression,
                classification, and clustering
              </li>
              <li>Accessibility text analysis powered by NLP models</li>
              <li>
                Interactive chart rendering with Matplotlib output served as
                static images
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-xl font-semibold text-primary mb-3">Outcome</h2>
            <p className="text-gray-600 text-sm leading-relaxed">
              df.jsx demonstrates how full-stack web technologies can be
              combined with Python-based analytics to deliver an end-to-end data
              science application. The project highlights integration patterns
              between a modern JavaScript frontend and a high-performance Python
              backend, making it possible to run and visualize machine learning
              models entirely in a browser-based UI.
            </p>
          </section>

          <div className="flex justify-center gap-4">
            <Link
              to="/predictors"
              className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-5 py-2.5 text-white text-sm font-semibold shadow-sm hover:bg-accent/90"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              Try df.jsx Now
            </Link>
            <Link
              to="/contact"
              className="inline-flex items-center justify-center gap-2 rounded-md border border-accent/30 px-5 py-2.5 text-accent text-sm font-semibold hover:bg-accent/10"
            >
              Share Feedback
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
