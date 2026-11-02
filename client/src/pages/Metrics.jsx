import { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from 'chart.js';
import { API_URL } from '../api';
import '../App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const pct = (n, d = 1) => (n == null ? '–' : `${Number(n).toFixed(d)}%`);

export default function Metrics() {
  const [data, setData] = useState(null);
  const [crops, setCrops] = useState([]);
  const [model, setModel] = useState('');
  const [importance, setImportance] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([fetch(`${API_URL}/metrics`).then(r => r.json()), fetch(`${API_URL}/metrics/commodities`).then(r => r.json())])
      .then(([m, c]) => { setData(m); setCrops(c); setModel(m.models.find(x => x.Type === 'Model' && x.Model !== 'Blend')?.Model || ''); })
      .catch(() => setError('Cannot reach the API. Is it running?'));
  }, []);

  useEffect(() => {
    if (!model) return;
    setImportance(null);
    fetch(`${API_URL}/feature-importance/${encodeURIComponent(model)}`)
      .then(r => r.json()).then(setImportance).catch(() => setImportance(null));
  }, [model]);

  if (!data) return <div className="app-container"><p>{error || 'Loading…'}</p></div>;

  const rows = data.models;
  const chart = {
    labels: rows.map(r => r.Model),
    datasets: [{ label: 'Mean absolute % error (lower is better)', data: rows.map(r => r.MAPE),
                 backgroundColor: rows.map(r => (r.Type === 'Baseline' ? '#9ca3af' : '#2e7d32')) }],
  };

  return (
    <div className="app-container">
      <header>
        <h1>Model Performance</h1>
        <p>
          Scored once on {data.rows.test.toLocaleString()} county-year observations from {data.test_years[0]}–{data.test_years[1]},
          years no model saw during training or tuning. Grey bars are simple baselines that use only earlier yields.
        </p>
      </header>

      <div className="main-content">
        <div className="result-card">
          <Bar data={chart} options={{ animation: false, indexAxis: 'y', plugins: { legend: { display: true } }, scales: { x: { beginAtZero: true } } }} />
        </div>

        <h2 style={{ marginTop: 32 }}>All models</h2>
        <div className="table-container">
          <table>
            <thead><tr><th>Model</th><th>Type</th><th>Mean error</th><th>Median error</th><th>Within 10%</th><th>R² (log)</th><th>Error removed vs best baseline</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.Model}>
                  <td><strong>{r.Model}</strong></td><td>{r.Type}</td><td>{pct(r.MAPE)}</td><td>{pct(r.MedAPE)}</td>
                  <td>{pct(r.within10, 0)}</td><td>{Number(r.R2_log).toFixed(3)}</td>
                  <td>{r.Type === 'Baseline' ? '–' : pct(100 * r.skill_vs_best_baseline, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="typical-range">
          Errors are percentages of the true yield, so crops measured in bushels, pounds and tons compare fairly.
          “Best baseline” is {data.best_baseline}.
        </p>

        <h2 style={{ marginTop: 32 }}>By crop ({data.best_model})</h2>
        <div className="table-container">
          <table>
            <thead><tr><th>Crop</th><th>Observations</th><th>Mean error</th><th>Median error</th><th>Baseline error</th><th>Error removed</th></tr></thead>
            <tbody>
              {crops.map(c => (
                <tr key={c.commodity}>
                  <td><strong>{c.commodity}</strong></td><td>{c.n.toLocaleString()}</td><td>{pct(c.MAPE)}</td><td>{pct(c.MedAPE)}</td>
                  <td>{pct(c.baseline_MAPE)}</td><td>{pct(100 * c.skill, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h2 style={{ marginTop: 32 }}>What drives a model’s predictions</h2>
        <div className="input-group">
          <select value={model} onChange={e => setModel(e.target.value)}>
            {rows.filter(r => r.Type === 'Model').map(r => <option key={r.Model} value={r.Model}>{r.Model}</option>)}
          </select>
        </div>
        {importance && importance.features.length > 0 ? (
          <div className="result-card">
            <Bar data={{ labels: importance.features.slice(0, 15).map(f => f.Feature),
                         datasets: [{ label: 'Importance', data: importance.features.slice(0, 15).map(f => f.Importance), backgroundColor: '#2e7d32' }] }}
                 options={{ animation: false, indexAxis: 'y', plugins: { legend: { display: false } } }} />
          </div>
        ) : <p className="typical-range">Feature importance isn’t available for this model.</p>}
      </div>
    </div>
  );
}
