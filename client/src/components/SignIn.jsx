// client/src/components/SignIn.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import newlogo500 from "../assets/newlogo500.png";

export default function SignIn({ setUser }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const logo = newlogo500;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      // 1) POST to /api/auth/jwt/login so the cookie is set
      const loginRes = await fetch("/api/auth/jwt/login", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          username: email,
          password: password,
        }).toString(),
      });

      if (!loginRes.ok) {
        const text = await loginRes.text();
        console.error("❌ Login failed:", loginRes.status, text);
        throw new Error("Login failed");
      }

      // 2) GET /api/users/me (cookie is sent automatically)
      const userRes = await fetch("/api/users/me", {
        credentials: "include",
      });
      if (!userRes.ok) {
        const text = await userRes.text();
        console.error("❌ /api/users/me failed:", userRes.status, text);
        throw new Error("Couldn’t load current user");
      }
      const userData = await userRes.json();

      // 3) Store user and update App state
      localStorage.setItem("user", JSON.stringify(userData));
      setUser(userData);

      // 4) Navigate
      navigate("/dashboard");
    } catch (err) {
      console.error("Login error:", err);
      setError("Invalid email or password");
    }
  };

  return (
    <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8 bg-cyan-50">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img alt="Your Company" src={logo} className="mx-auto h-48 w-auto" />
        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-white">
          Sign in to your account
        </h2>
      </div>

      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-blue-800"
            >
              Email address
            </label>
            <div className="mt-2">
              <input
                id="email"
                name="email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="block w-full rounded-md bg-pink/5 px-3 py-1.5 text-black outline outline-1 -outline-offset-1 outline-white/10 placeholder:text-gray-500 focus:outline focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-500 sm:text-sm"
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-blue-800"
            >
              Password
            </label>
            <div className="mt-2">
              <input
                id="password"
                name="password"
                type="password"
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full rounded-md bg-pink/5 px-3 py-1.5 text-black outline outline-1 -outline-offset-1 outline-white/10 placeholder:text-gray-500 focus:outline focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-500 sm:text-sm"
              />
            </div>
          </div>

          {error && <p className="text-red-400 text-sm">{error}</p>}

          <div>
            <button
              type="submit"
              className="flex w-full justify-center rounded-md bg-blue-800 px-3 py-1.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:outline focus:outline-2 focus:outline-offset-2 focus:outline-indigo-500"
            >
              Sign in
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
