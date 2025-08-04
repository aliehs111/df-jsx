// src/App.jsx
import "./App.css";
import React, { useEffect, useState } from "react";
import {
  HashRouter as Router,
  Routes,
  Route,
  Navigate,
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

function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  useEffect(() => {
    const fetchCurrentUser = async () => {
      try {
        const res = await fetch("/api/users/me", {
          credentials: "include",
        });
        if (res.ok) {
          const userData = await res.json();
          setUser(userData);
        }
      } catch {
        // if network error or 401, remain unauthenticated
      } finally {
        setCheckingAuth(false);
      }
    };
    fetchCurrentUser();
  }, []);

  if (checkingAuth) {
    return <div className="text-center py-8">Loading...</div>;
  }

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

      {/* 👇 Add Databot floating widget outside Routes */}
      {user && <Databot />}

      <Footer />
    </Router>
  );
}

export default App;
