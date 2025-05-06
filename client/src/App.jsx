// src/App.jsx
import "./App.css";
import React, { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import axios from "axios";

import SignIn from "./components/SignIn";
import SignUp from "./components/SignUp"; // ← import your new signup
import FileUpload from "./components/FileUpload";
import DataCleaning from "./components/DataCleaning";
import Dashboard from "./components/Dashboard";
import DatasetsList from "./components/DatasetsList";
import DatasetDetail from "./components/DatasetDetail";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Splash from "./components/Splash";

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setLoading(false);
      return;
    }

    axios
      .get("/users/me", {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((res) => setUser(res.data))
      .catch((err) => {
        console.error("Auth error:", err);
        localStorage.removeItem("token");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading...</div>;

  const ProtectedRoute = ({ element }) =>
    user ? element : <Navigate to="/login" replace />;

  return (
    <Router>
      <Navbar user={user} setUser={setUser} />

      <Routes>
        {/* public routes */}
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

        {/* protected */}
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

        {/* catch-all: if logged in → dashboard, otherwise show splash */}
        <Route
          path="*"
          element={user ? <Navigate to="/dashboard" replace /> : <Splash />}
        />
      </Routes>

      <Footer />
    </Router>
  );
}

export default App;
