import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Predict from './pages/Predict';
import Home from './pages/Home';
import Metrics from './pages/Metrics';
import Regional from './pages/Regional';
import './App.css';

function NotFound() {
  return (
    <div className="app-container">
      <header><h1>Page not found</h1><p>That address doesn't exist. Try <Link to="/predict">Predict</Link> or <Link to="/">Home</Link>.</p></header>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <nav className="navbar">
          <div className="nav-logo">🌱 Crop Yield Predictor</div>
          <ul className="nav-links">
            <li><Link to="/">Home</Link></li>
            <li><Link to="/predict">Predict</Link></li>
            <li><Link to="/regional">Regional</Link></li>
            <li><Link to="/metrics">Metrics</Link></li>
          </ul>
        </nav>

        <div className="page-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/regional" element={<Regional />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
