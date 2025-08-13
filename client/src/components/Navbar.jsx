// src/components/Navbar.jsx
import { NavLink, useNavigate } from "react-router-dom";
import { Disclosure } from "@headlessui/react";
import {
  Bars3Icon,
  XMarkIcon,
  ArrowRightOnRectangleIcon,
  UserPlusIcon,
} from "@heroicons/react/24/outline";
import newlogo500 from "../assets/newlogo500.png";

const NAV_LINKS = [
  { name: "Dashboard", to: "/dashboard" },
  { name: "Upload", to: "/upload" },
  { name: "Datasets", to: "/datasets" },
  { name: "Resources", to: "/resources" },
  { name: "Models", to: "/models" },
  { name: "Predictors", to: "/predictors" },
  { name: "About", to: "/about" },
];

export default function Navbar({ user, setUser }) {
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/jwt/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      setUser?.(null);
      navigate("/", { replace: true });
    }
  };

  const desktopLinkClasses = ({ isActive }) =>
    [
      "relative inline-flex items-center px-3 py-2 text-sm font-medium transition-colors",
      isActive
        ? "text-accent border-b-2 border-accent"
        : "text-white hover:text-accent",
    ].join(" ");

  const mobileLinkClasses = ({ isActive }) =>
    [
      "block w-full rounded-md px-3 py-2 text-base font-medium",
      isActive ? "bg-accent text-white" : "text-white hover:bg-primary/80",
    ].join(" ");

  return (
    <Disclosure
      as="nav"
      className="sticky top-0 z-50 bg-primary text-white shadow-lg"
    >
      {({ open }) => (
        <>
          {/* Desktop */}
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 items-center justify-between">
              {/* Logo + links */}
              <div className="flex items-center gap-6">
                <NavLink to="/" className="inline-flex items-center">
                  <img
                    src={newlogo500}
                    alt="df.jsx logo"
                    className="h-9 w-9 block rounded-md shadow-sm shrink-0"
                  />
                </NavLink>

                <div className="hidden sm:flex sm:items-center sm:gap-1">
                  {NAV_LINKS.map((link) => (
                    <NavLink
                      key={link.name}
                      to={link.to}
                      className={desktopLinkClasses}
                    >
                      {link.name}
                    </NavLink>
                  ))}
                </div>
              </div>

              {/* Auth controls */}
              <div className="hidden sm:flex sm:items-center sm:gap-3">
                {user ? (
                  <>
                    <span className="text-sm">
                      Welcome,{" "}
                      <span className="font-semibold">{user.email}</span>
                    </span>
                    <button
                      onClick={handleLogout}
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white shadow-sm transition hover:bg-accent/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50"
                    >
                      <ArrowRightOnRectangleIcon className="h-5 w-5" />
                      Logout
                    </button>
                  </>
                ) : (
                  <>
                    <NavLink
                      to="/login"
                      className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-semibold text-white hover:text-accent"
                    >
                      <ArrowRightOnRectangleIcon className="h-5 w-5" />
                      Login
                    </NavLink>
                    <NavLink
                      to="/signup"
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white shadow-sm transition hover:bg-accent/90"
                    >
                      <UserPlusIcon className="h-5 w-5" />
                      Sign Up
                    </NavLink>
                  </>
                )}
              </div>

              {/* Mobile button */}
              <div className="flex items-center sm:hidden">
                <Disclosure.Button className="inline-flex items-center justify-center rounded-md p-2 text-white hover:bg-primary/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50">
                  <span className="sr-only">Open main menu</span>
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
          <Disclosure.Panel className="sm:hidden bg-primary/95">
            <div className="space-y-1 px-3 py-3">
              {NAV_LINKS.map((link) => (
                <Disclosure.Button
                  key={link.name}
                  as={NavLink}
                  to={link.to}
                  className={mobileLinkClasses}
                >
                  {link.name}
                </Disclosure.Button>
              ))}

              <div className="mt-2 border-t border-white/20 pt-2">
                {user ? (
                  <Disclosure.Button
                    as="button"
                    onClick={handleLogout}
                    className="flex w-full items-center gap-2 rounded-md bg-accent px-3 py-2 text-base font-medium text-white hover:bg-accent/90"
                  >
                    <ArrowRightOnRectangleIcon className="h-6 w-6" />
                    Logout
                  </Disclosure.Button>
                ) : (
                  <div className="flex flex-col gap-2">
                    <Disclosure.Button
                      as={NavLink}
                      to="/login"
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-base font-medium text-white hover:text-accent hover:bg-primary/80"
                    >
                      <ArrowRightOnRectangleIcon className="h-6 w-6" />
                      Login
                    </Disclosure.Button>
                    <Disclosure.Button
                      as={NavLink}
                      to="/signup"
                      className="flex w-full items-center gap-2 rounded-md bg-accent px-3 py-2 text-base font-medium text-white hover:bg-accent/90"
                    >
                      <UserPlusIcon className="h-6 w-6" />
                      Sign Up
                    </Disclosure.Button>
                  </div>
                )}
              </div>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  );
}
