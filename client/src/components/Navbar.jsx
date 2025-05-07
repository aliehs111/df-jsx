// src/components/Navbar.jsx
import { NavLink, useNavigate } from 'react-router-dom'
import { Disclosure } from '@headlessui/react'
import {
  Bars3Icon,
  XMarkIcon,
  ArrowRightOnRectangleIcon,
  UserPlusIcon,
  BellIcon,
} from '@heroicons/react/24/outline'
import newlogo500 from '../assets/newlogo500.png'

const NAV_LINKS = [
  { name: 'Dashboard',    to: '/dashboard' },
  { name: 'Upload',       to: '/upload' },
  { name: 'My Datasets',  to: '/datasets' },
  { name: 'Resources',    to: '/resources'},
]

export default function Navbar({ user, setUser }) {
  const navigate = useNavigate()

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)                // ← clear App’s user state
    navigate('/', { replace: true })
  }

  return (
    <Disclosure as="nav" className="bg-cyan-400 shadow">
      {({ open }) => (
        <>
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 justify-between">
              
              {/* Logo + Links */}
              <div className="flex ">
                <NavLink to="/dashboard" className="flex items-center">
                  <img src={newlogo500} alt="logo" className="h-12 w-auto rounded-md" />
                </NavLink>
                <div className="hidden sm:ml-8 sm:flex sm:space-x-4">
                  {NAV_LINKS.map((link) => (
                    <NavLink
                      key={link.name}
                      to={link.to}
                      className={({ isActive }) =>
                        (isActive
                          ? 'border-b-2 border-indigo-700 text-indigo-900'
                          : 'text-white hover:text-gray-100') +
                        ' px-3 py-2 rounded-md text-sm font-medium'
                      }
                    >
                      {link.name}
                    </NavLink>
                  ))}
                </div>
              </div>

              {/* Auth/UI controls */}
              <div className="hidden sm:flex sm:items-center sm:space-x-4">
                {user ? (
                  <>
             
                    <span className="text-md text-indigo-900">
                      Welcome, <span className="font-semibold">{user.email}</span>
                    </span>
                    <button
                      onClick={handleLogout}
                      className="flex items-center space-x-1 bg-blue-800 hover:bg-lime-400 px-3 py-1 rounded-md text-sm text-white"
                    >
                      <ArrowRightOnRectangleIcon className="h-5 w-5" />
                      <span>Logout</span>
                    </button>
                  </>
                ) : (
                  <>
                    <NavLink
                      to="/login"
                      className="flex items-center space-x-1 text-white hover:underline text-sm"
                    >
                      <ArrowRightOnRectangleIcon className="h-5 w-5" />
                      <span>Login</span>
                    </NavLink>
                    <NavLink
                      to="/signup"
                      className="flex items-center space-x-1 bg-blue-800 hover:bg-green-800 px-3 py-1 rounded-md text-sm text-white"
                    >
                      <UserPlusIcon className="h-5 w-5" />
                      <span>Sign Up</span>
                    </NavLink>
                  </>
                )}
              </div>

              {/* Mobile menu button */}
              <div className="-mr-2 flex items-center sm:hidden">
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-white hover:bg-indigo-500">
                  {open ? (
                    <XMarkIcon className="h-6 w-6" />
                  ) : (
                    <Bars3Icon className="h-6 w-6" />
                  )}
                </Disclosure.Button>
              </div>
            </div>
          </div>

          {/* Mobile menu */}
          <Disclosure.Panel className="sm:hidden bg-cyan-200">
            <div className="space-y-1 px-2 pt-2 pb-3">
              {NAV_LINKS.map((link) => (
                <Disclosure.Button
                  key={link.name}
                  as={NavLink}
                  to={link.to}
                  className={({ isActive }) =>
                    (isActive
                      ? 'bg-indigo-700 text-white'
                      : 'text-indigo-900 hover:bg-indigo-300') +
                    ' block px-3 py-2 rounded-md text-base font-medium'
                  }
                >
                  {link.name}
                </Disclosure.Button>
              ))}

              <div className="border-t border-indigo-300/50 mt-2 pt-2">
                {user ? (
                  <Disclosure.Button
                    as="button"
                    onClick={handleLogout}
                    className="flex w-full items-center space-x-2 px-3 py-2 rounded-md text-indigo-900 hover:bg-indigo-300"
                  >
                    <ArrowRightOnRectangleIcon className="h-6 w-6" />
                    <span>Logout</span>
                  </Disclosure.Button>
                ) : (
                  <>
                    <Disclosure.Button
                      as={NavLink}
                      to="/login"
                      className="flex w-full items-center space-x-2 px-3 py-2 rounded-md text-indigo-900 hover:bg-indigo-300"
                    >
                      <ArrowRightOnRectangleIcon className="h-6 w-6" />
                      <span>Login</span>
                    </Disclosure.Button>
                    <Disclosure.Button
                      as={NavLink}
                      to="/signup"
                      className="flex w-full items-center space-x-2 px-3 py-2 rounded-md bg-indigo-700 text-white hover:bg-indigo-800"
                    >
                      <UserPlusIcon className="h-6 w-6" />
                      <span>Sign Up</span>
                    </Disclosure.Button>
                  </>
                )}
              </div>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  )
}


