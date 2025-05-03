import './App.css';
import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

import SignIn from './components/SignIn';
import FileUpload from './components/FileUpload';
import DataCleaning from './components/DataCleaning';
import Dashboard from './components/Dashboard';
import DatasetsList from './components/DatasetsList';
import DatasetDetail from './components/DatasetDetail';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

function App() {
  const [user, setUser] = useState(null);  // holds user data from /users/me
  const [loading, setLoading] = useState(true);  // waits for token check

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setLoading(false);
      return;
    }

    axios.get('/users/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    .then((res) => {
      setUser(res.data);
    })
    .catch((err) => {
      console.error("Auth error:", err);
      localStorage.removeItem('token');
    })
    .finally(() => {
      setLoading(false);
    });
  }, []);

  if (loading) return <div>Loading...</div>;

  const ProtectedRoute = ({ element }) =>
    user ? element : <Navigate to="/" />;

  return (
    <Router>
    <Navbar />
      <Routes>
        <Route path="/" element={<SignIn setUser={setUser} />} />
        <Route path="/upload" element={<ProtectedRoute element={<FileUpload user={user} />} />} />
        <Route path="/datasets/:id/clean" element={<ProtectedRoute element={<DataCleaning />} />} />
        <Route path="/dashboard" element={<ProtectedRoute element={<Dashboard user={user} />} />} />
        <Route path="/datasets" element={<ProtectedRoute element={<DatasetsList />} />} />
        <Route path="/datasets/:id" element={<ProtectedRoute element={<DatasetDetail />} />} />
      </Routes>
      <Footer /> 
    </Router>
  );
}

export default App;
