import './App.css'
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SignIn from './components/SignIn';
import FileUpload from './components/FileUpload';
import DataCleaning from './components/DataCleaning';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SignIn />} />
        <Route path="/upload" element={<FileUpload />} />
        <Route path="/clean" element={<DataCleaning />} />
        <Route path="/dashboard" element={<Dashboard />} />
     
      </Routes>
    </Router>
  );
}

export default App;





