import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { API_URL } from '../api';
import '../App.css';

export default function Home() {
  const navigate = useNavigate();
  const [meta, setMeta] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/metadata`).then(r => r.json()).then(setMeta).catch(() => {});
    fetch(`${API_URL}/metrics`).then(r => r.json()).then(setMetrics).catch(() => {});
  }, []);

  const best = metrics?.models?.find(m => m.Model === metrics.best_model);
  const baseline = metrics?.models?.find(m => m.Model === metrics.best_baseline);
  const seriesCount = meta?.crops?.reduce((n, c) => n + c.series.length, 0);

  return (
    <div className="home-container">
      <div className="hero-section">
        <h1 className="hero-title">🌱 Crop Yield Prediction</h1>
        <p className="hero-subtitle">
          County-level yield predictions for US crops, from NASS yield records, NASA POWER weather,
          NOAA drought indices and SSURGO soil data
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">🌾</div>
          <div className="stat-value">{meta ? meta.crops.length : '–'}</div>
          <div className="stat-label">Crops</div>
          <div className="stat-detail">{seriesCount ? `${seriesCount} yield series (class × practice)` : ''}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🤖</div>
          <div className="stat-value">{meta ? meta.models.length : '–'}</div>
          <div className="stat-label">Models</div>
          <div className="stat-detail">Default: {meta?.default_model}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-value">{best ? `${best.MedAPE.toFixed(1)}%` : '–'}</div>
          <div className="stat-label">Median error</div>
          <div className="stat-detail">{metrics ? `on held-out ${metrics.test_years[0]}–${metrics.test_years[1]}` : ''}</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📉</div>
          <div className="stat-value">{best && baseline ? `${Math.round(100 * (1 - best.MAPE / baseline.MAPE))}%` : '–'}</div>
          <div className="stat-label">Less error than persistence</div>
          <div className="stat-detail">vs. "same as the last observed yield"</div>
        </div>
      </div>

      <div className="features-section">
        <h2>What you can do</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Predict a county</h3>
            <p>Pick a crop, variety (class and production practice), county and year. Weather and soil default to the county's own normals and can be overridden.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🌍</div>
            <h3>Compare crops</h3>
            <p>See predictions for every crop a county grows, side by side.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔬</div>
            <h3>Check the evidence</h3>
            <p>Every model is scored on years it never saw and compared with simple baselines, per crop.</p>
          </div>
        </div>
      </div>

      <div className="cta-section">
        <button className="cta-button primary" onClick={() => navigate('/predict')}>Start predicting →</button>
        <button className="cta-button secondary" onClick={() => navigate('/metrics')}>View model performance</button>
      </div>
    </div>
  );
}
