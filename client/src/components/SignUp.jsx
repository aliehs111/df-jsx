// src/components/SignUp.jsx
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";

export default function SignUp() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        const payload = await res.json();
        throw new Error(
          payload.detail ||
            `Registration failed: ${res.status} ${res.statusText}`
        );
      }

      navigate("/login");
    } catch (err) {
      setError(err.message);
      console.error("Registration error:", err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 py-12 px-4">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center">
          <img
            src={newlogo500}
            alt="Logo"
            className="mx-auto h-48 w-auto rounded-md"
          />
          <h2 className="mt-6 text-3xl font-bold text-blue-800">
            Create your account
          </h2>
          <p className="mt-2 text-sm text-blue-800">
            Already have an account?{" "}
            <Link
              to="/login"
              className="font-medium text-cyan-600 hover:text-cyan-500"
            >
              Sign in
            </Link>
          </p>
        </div>

        <div className="bg-white py-8 px-6 shadow rounded-lg">
          {error && <div className="mb-4 text-red-600 text-sm">{error}</div>}
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email address
              </label>
              <input
                id="email"
                type="email"
                required
                disabled={loading}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm px-3 py-2 focus:ring-cyan-500 focus:border-cyan-500 sm:text-sm"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                disabled={loading}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm px-3 py-2 focus:ring-cyan-500 focus:border-cyan-500 sm:text-sm"
              />
            </div>

            <div>
              <label
                htmlFor="confirm"
                className="block text-sm font-medium text-gray-700"
              >
                Confirm password
              </label>
              <input
                id="confirm"
                type="password"
                required
                disabled={loading}
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm px-3 py-2 focus:ring-cyan-500 focus:border-cyan-500 sm:text-sm"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white ${
                loading
                  ? "bg-cyan-300 cursor-not-allowed"
                  : "bg-cyan-600 hover:bg-cyan-700"
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500`}
            >
              {loading ? "Creating…" : "Sign Up"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
