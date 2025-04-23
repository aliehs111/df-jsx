import { Link, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'

export default function Navbar() {
  const [userEmail, setUserEmail] = useState(null)
  const navigate = useNavigate()

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
    <nav className="bg-indigo-600 text-white px-4 py-3 flex justify-between items-center shadow-sm">
      <div className="text-lg font-semibold">
        <Link to="/dashboard">dfjsx</Link>
      </div>
      <div className="flex gap-6 items-center text-sm font-medium">
        <Link to="/upload" className="hover:underline">Upload</Link>
        <Link to="/datasets" className="hover:underline">My Datasets</Link>
        {userEmail && <span className="text-gray-200">Welcome, {userEmail}</span>}
        <button onClick={handleLogout} className="bg-indigo-500 hover:bg-indigo-400 px-3 py-1 rounded text-white">Logout</button>
      </div>
    </nav>
  )
}
