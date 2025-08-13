// src/components/Splash.jsx
import { Link } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";

export default function Splash() {
  return (
    <div className="w-full bg-gradient-to-b from-neutralLight to-white px-4 py-16">
      <div className="mx-auto max-w-2xl text-center rounded-3xl bg-white shadow-xl ring-1 ring-black/5 px-8 py-12 md:px-12 md:py-14">
        {/* Logo (kept large) */}
        <img
          src={newlogo500}
          alt="df.jsx Logo"
          className="h-44 w-44 sm:h-48 sm:w-48 mx-auto mb-6 rounded-2xl shadow-md"
        />

        {/* Heading */}
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-primary">
          Welcome to df.jsx
        </h1>

        {/* Subtext */}
        <p className="mt-3 text-base sm:text-lg leading-relaxed text-neutralDark/80 max-w-xl mx-auto">
          An academic experiment to develop an application, exploring ways to
          teach about data analysis and machine learning with AI-driven
          guidance, Databot, a tutor-style chatbot.
        </p>

        {/* Actions */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            to="/login"
            className="px-6 py-3 rounded-lg bg-primary text-white font-semibold shadow-md hover:bg-secondary hover:shadow-lg transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary"
          >
            Log In
          </Link>

          <Link
            to="/signup"
            className="px-6 py-3 rounded-lg border-2 border-primary text-primary font-semibold hover:bg-accent hover:border-accent hover:text-white transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary"
          >
            Sign Up
          </Link>
        </div>
      </div>
    </div>
  );
}
