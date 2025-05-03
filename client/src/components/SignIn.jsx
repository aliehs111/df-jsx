// client/src/components/SignIn.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
  
    try {
      // Step 1: Login to get token (relative URL)
      const response = await fetch('/auth/jwt/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: email,
          password: password,
        }).toString(),
      });
  
      console.log("STATUS:", response.status);
      console.log("HEADERS:", Array.from(response.headers.entries()));
  
      const data = await response.json().catch((err) => {
        console.error("❌ Failed to parse JSON:", err);
        return null;
      });
      console.log("✅ Login response data:", data);
  
      if (!response.ok || !data) {
        throw new Error('Login failed');
      }
  
      const token = data.access_token;
      localStorage.setItem('token', token);
  
      // Step 2: Get current user info (also relative)
      const userRes = await fetch('/users/me', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
  
      const userData = await userRes.json();
      localStorage.setItem('user', JSON.stringify(userData));
  
      // Step 3: Redirect to dashboard
      navigate('/dashboard');
  
    } catch (err) {
      console.error('Login error:', err);
      setError('Invalid email or password');
    }
  };
  
  return (
    <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-sm">
        <img
          alt="Your Company"
          src="https://tailwindcss.com/plus-assets/img/logos/mark.svg?color=indigo&shade=500"
          className="mx-auto h-10 w-auto"
        />
        <h2 className="mt-10 text-center text-2xl font-bold tracking-tight text-white">
          Sign in to your account
        </h2>
      </div>

      <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-blacl">
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
            <label htmlFor="password" className="block text-sm font-medium text-black">
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
              className="flex w-full justify-center rounded-md bg-indigo-500 px-3 py-1.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-400 focus:outline focus:outline-2 focus:outline-offset-2 focus:outline-indigo-500"
            >
              Sign in
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

