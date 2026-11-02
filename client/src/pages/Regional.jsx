import { useEffect, useState } from 'react';
import { API_URL } from '../api';
import '../App.css';

const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 });

export default function Regional() {
  const [meta, setMeta] = useState(null);
  const [states, setStates] = useState([]);
  const [counties, setCounties] = useState([]);
  const [form, setForm] = useState({ state: '', county: '', year: new Date().getFullYear() });
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/metadata`).then(r => r.json()).then(m => {
      setMeta(m);
      setStates([...new Set(m.crops.flatMap(c => c.states))].sort());
    }).catch(() => setError('Cannot reach the API. Is it running?'));
  }, []);

  useEffect(() => {
    if (!form.state) { setCounties([]); return; }
    fetch(`${API_URL}/counties/${form.state}`).then(r => r.json()).then(d => setCounties(d.counties || [])).catch(() => setCounties([]));
  }, [form.state]);

  async function submit(e) {
    e.preventDefault();
    setLoading(true); setError(null); setOut(null);
    try {
      const res = await fetch(`${API_URL}/predict-regional`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: form.state, county: form.county, year: Number(form.year) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      setOut(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!meta) return <div className="app-container"><p>{error || 'Loading…'}</p></div>;

  return (
    <div className="app-container">
      <header>
        <h1>🌱 Regional Crop Comparison</h1>
        <p>Every crop a county grows, predicted for one year</p>
      </header>
      <div className="main-content">
        <form onSubmit={submit} className="prediction-form">
          <div className="form-section">
            <div className="input-group">
              <label>State</label>
              <select name="State" value={form.state} onChange={e => setForm({ ...form, state: e.target.value, county: '' })} required>
                <option value="">Select a state</option>
                {states.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label>County</label>
              <select name="County" value={form.county} onChange={e => setForm({ ...form, county: e.target.value })} required disabled={!counties.length}>
                <option value="">{form.state ? 'Select a county' : 'Choose a state first'}</option>
                {counties.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label>Year</label>
              <input type="number" name="Year" min="1981" max="2100" value={form.year} onChange={e => setForm({ ...form, year: e.target.value })} required />
            </div>
          </div>
          <button type="submit" className="submit-btn" disabled={loading || !form.county}>{loading ? 'Predicting…' : 'Predict all crops'}</button>
        </form>

        {error && <div className="error-msg">{error}</div>}

        {out && (
          <div className="result-card">
            <h2>{form.county}, {form.state} · {out.year}</h2>
            <div className="table-container">
              <table>
                <thead><tr><th>Crop</th><th>Predicted</th><th>Unit</th><th>Last observed</th><th>Typical error</th></tr></thead>
                <tbody>
                  {out.results.map(r => (
                    <tr key={r.series}>
                      <td><strong>{r.commodity}</strong></td>
                      <td>{fmt(r.prediction)}</td>
                      <td>{r.unit.toLowerCase()}</td>
                      <td>{fmt(r.last_observed.yield)} ({r.last_observed.year})</td>
                      <td>{r.expected_error_pct != null ? `±${r.expected_error_pct}%` : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {out.not_grown.length > 0 && (
              <p className="typical-range">No yield records here for: {out.not_grown.map(c => c.toLowerCase()).join(', ')}.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
