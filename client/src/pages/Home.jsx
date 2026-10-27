import React, { useState, useEffect } from 'react';
import { API_URL } from '../api';
import { useNavigate } from 'react-router-dom';
import '../App.css';

export default function Home() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    models: 15,
    crops: 0,
    states: 0,
    bestR2: 0.987
  });

  useEffect(() => {
    // Fetch metadata to get actual crop and state counts
    fetch(`${API_URL}/metadata`)
      .then(res => res.json())
      .then(data => {
        setStats(prev => ({
          ...prev,
          crops: data.crops?.length || 11,
          states: data.states?.length || 50 
        }));
      })
      .catch(err => console.error('Failed to fetch metadata', err));
  }, []);

  return (
    <div className="home-container">
      <div className="hero-section">
        <h1 className="hero-title">🌱 Crop Yield Prediction Platform</h1>
        <p className="hero-subtitle">
          Advanced machine learning models for agricultural yield forecasting using climate, soil, and geospatial data
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">🤖</div>
          <div className="stat-value">{stats.models}</div>
          <div className="stat-label">ML Models</div>
          <div className="stat-detail">4 Baselines + 5 Classical + 6 Ensemble</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🌾</div>
          <div className="stat-value">{stats.crops}</div>
          <div className="stat-label">Crop Types</div>
          <div className="stat-detail">Corn, Wheat, Soybeans & More</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🗺️</div>
          <div className="stat-value">{stats.states}</div>
          <div className="stat-label">States Covered</div>
          <div className="stat-detail">Nationwide Coverage</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">⭐</div>
          <div className="stat-value">{stats.bestR2}</div>
          <div className="stat-label">Best R² Score</div>
          <div className="stat-detail">Random Forest Model</div>
        </div>
      </div>

      <div className="features-section">
        <h2>Key Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Single Crop Prediction</h3>
            <p>Get precise yield predictions for individual crops based on weather, soil, and location data</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🌍</div>
            <h3>Regional Analysis</h3>
            <p>Compare yield predictions across all crops for a specific region</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🔬</div>
            <h3>Model Comparison</h3>
            <p>Evaluate trade-offs between accuracy, speed, and interpretability across multiple ML models</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">💡</div>
            <h3>Feature Insights</h3>
            <p>Understand which factors (weather, soil, etc.) drive yield predictions</p>
          </div>
        </div>
      </div>

      <div className="cta-section">
        <button className="cta-button primary" onClick={() => navigate('/predict')}>
          Start Predicting →
        </button>
        <button className="cta-button secondary" onClick={() => navigate('/metrics')}>
          View Model Performance
        </button>
      </div>
    </div>
  );
}
