import "./App.css";
import React, { useEffect, useState } from "react";
import {
  HashRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from "react-router-dom";

import SignIn from "./components/SignIn";
import SignUp from "./components/SignUp";
import FileUpload from "./components/FileUpload";
import DataCleaning from "./components/DataCleaning";
import Dashboard from "./components/Dashboard";
import DatasetsList from "./components/DatasetsList";
import DatasetDetail from "./components/DatasetDetail";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Splash from "./components/Splash";
import Resources from "./components/Resources";
import ProcessDataset from "./components/ProcessDataset";
import DataInsights from "./components/DataInsights";
import Models from "./components/Models";
import Databot from "./components/Databot";
import Predictors from "./components/Predictors";
import About from "./components/About";

function DatabotWrapper({ user }) {
  const location = useLocation();
  if (
    ["/upload", "/datasets", "/resources", "/about"].includes(location.pathname)
  ) {
    return null;
  }
  return <Databot />;
}

function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  // first effect: initial check (keep as is)
  useEffect(() => {
    const fetchCurrentUser = async () => {
      try {
        const res = await fetch("/api/users/me", { credentials: "include" });
        if (res.ok) {
          const userData = await res.json();
          setUser(userData);
        }
      } catch {
        // stay unauthenticated
      } finally {
        setCheckingAuth(false);
      }
    };
    fetchCurrentUser();
  }, []);

  // second effect: re-check on return from idle
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch("/api/users/me", { credentials: "include" });
        if (!res.ok) {
          setUser(null);
          window.location.hash = "#/"; // go back to splash
        }
      } catch {
        setUser(null);
        window.location.hash = "#/";
      }
    };

    const onFocus = () => checkAuth();
    const onVisible = () => {
      if (document.visibilityState === "visible") checkAuth();
    };

    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const ProtectedRoute = ({ element }) =>
    user ? element : <Navigate to="/" replace />;

  return (
    <Router>
      <Navbar user={user} setUser={setUser} />

      <Routes>
        {/* Public routes */}
        <Route
          path="/"
          element={user ? <Navigate to="/dashboard" replace /> : <Splash />}
        />
        <Route
          path="/login"
          element={
            user ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <SignIn setUser={setUser} />
            )
          }
        />
        <Route
          path="/signup"
          element={
            user ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <SignUp setUser={setUser} />
            )
          }
        />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={<ProtectedRoute element={<Dashboard user={user} />} />}
        />
        <Route
          path="/upload"
          element={<ProtectedRoute element={<FileUpload user={user} />} />}
        />
        <Route
          path="/datasets"
          element={<ProtectedRoute element={<DatasetsList />} />}
        />
        <Route
          path="/datasets/:id"
          element={<ProtectedRoute element={<DatasetDetail />} />}
        />
        <Route
          path="/datasets/:id/clean"
          element={<ProtectedRoute element={<DataCleaning />} />}
        />
        <Route
          path="/datasets/:id/insights"
          element={<ProtectedRoute element={<DataInsights />} />}
        />
        <Route
          path="/datasets/:id/process"
          element={<ProtectedRoute element={<ProcessDataset />} />}
        />
        <Route
          path="/resources"
          element={<ProtectedRoute element={<Resources />} />}
        />
        <Route
          path="/models"
          element={<ProtectedRoute element={<Models />} />}
        />
        <Route
          path="/predictors"
          element={<ProtectedRoute element={<Predictors />} />}
        />
        <Route path="/about" element={<ProtectedRoute element={<About />} />} />

        {/* Catch-all */}
        <Route
          path="*"
          element={
            user ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
      </Routes>

      {/* Add Databot floating widget inside Router */}
      {user && <DatabotWrapper user={user} />}

      <Footer />
    </Router>
  );
}

export default App;
