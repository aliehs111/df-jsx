// src/components/Splash.jsx
import { Link } from 'react-router-dom'
import newlogo500 from '../assets/newlogo500.png'

export default function Splash() {
  return (
    <div className="flex flex-col items-center justify-center bg-cyan-100 px-4">
      <img
        src={newlogo500}
        alt="App Logo"
        className="h-48 w-48 mb-4 rounded-md"
      />
      <h1 className="text-4xl font-blue-800 font-bold text-cyan-800">Welcome to df.jsx!</h1>
      <p className="mt-2 text-md text-cyan-700">
      Master Data Prep with Your Personal Chatbot Data Coach
      </p>

      <div className="mt-8 flex space-x-4">
        <Link
          to="/login"
          className="px-6 py-2 bg-blue-800 text-white rounded hover:bg-indigo-500 transition"
        >
          Log In
        </Link>
        <Link
          to="/signup"
          className="px-6 py-2 bg-blue-800 text-white rounded hover:bg-green-500 transition"
        >
          Sign Up
        </Link>
      </div>
    </div>
  )
}

  