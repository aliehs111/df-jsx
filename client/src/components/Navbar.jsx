import { Link, useNavigate, NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'

import logo512 from '../assets/logo512.png'
import newlogo500 from '../assets/newlogo500.png'

export default function Navbar() {
  const [userEmail, setUserEmail] = useState(null)
  const navigate = useNavigate()

const Navlogo = logo512
const NewLogo = newlogo500

  useEffect(() => {
    const storedUser = localStorage.getItem('user')
    if (storedUser) {
      const user = JSON.parse(storedUser)
      setUserEmail(user.email)
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/')
  }

  return (
    <nav className="bg-cyan-300 text-white px-4 py-3 flex justify-between items-center shadow-sm">
      <div className="flex items-center space-x-2">
        <NavLink to="/dashboard">
          <img
            src={NewLogo}
            alt="dfjsx logo"
            className="h-16 w-16 object-contain rounded-md"
          />
        </NavLink>
      </div>
      <div className="flex gap-6 items-center text-sm font-medium">
        <NavLink to="/upload" className={({ isActive }) => isActive ? "underline" : "hover:underline"}>
          Upload
        </NavLink>
        <NavLink to="/datasets" className={({ isActive }) => isActive ? "underline" : "hover:underline"}>
          My Datasets
        </NavLink>
        {userEmail && <span className="text-blue-700">Welcome, {userEmail}</span>}
        <button onClick={handleLogout} className="bg-lime-500 hover:bg-indigo-400 px-3 py-1 rounded text-white">
          Logout
        </button>
      </div>
    </nav>
  );
}
