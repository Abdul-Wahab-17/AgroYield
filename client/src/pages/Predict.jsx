import { useEffect, useMemo, useState } from 'react';
import { API_URL } from '../api';
import '../App.css';

const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 });
const label = (s) => (s.class === 'ALL CLASSES' ? '' : `${s.class} · `) + s.practice.replace('ALL PRODUCTION PRACTICES', 'all practices').toLowerCase();

const OVERRIDES = [
  ['Season weather', [['tmax_season', 'Avg max temp (°C)'], ['tmin_season', 'Avg min temp (°C)'], ['prcp_season', 'Season rainfall (mm)'],
    ['gdd_season', 'Growing degree days'], ['heat_days', 'Days above 35 °C']]],
  ['Soil (0–30 cm)', [['clay_pct', 'Clay (%)'], ['sand_pct', 'Sand (%)'], ['organic_matter_pct', 'Organic matter (%)'], ['ph', 'pH']]],
];

export default function Predict() {
  const [meta, setMeta] = useState(null);
  const [form, setForm] = useState({ commodity: 'CORN', series: '', state: '', county: '', year: new Date().getFullYear(), model: '' });
  const [overrides, setOverrides] = useState({});
  const [states, setStates] = useState([]);
  const [counties, setCounties] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/metadata`).then(r => r.json()).then(m => {
      setMeta(m);
      setForm(f => ({ ...f, model: m.default_model }));
    }).catch(() => setError('Cannot reach the API. Is it running?'));
  }, []);

  const crop = useMemo(() => meta?.crops.find(c => c.commodity === form.commodity), [meta, form.commodity]);
  const series = useMemo(() => crop?.series.find(s => s.series === form.series) || crop?.series.find(s => s.headline) || crop?.series[0], [crop, form.series]);

  useEffect(() => {
    if (!series) return;
    setStates(series.states);
    setForm(f => (f.state && !series.states.includes(f.state) ? { ...f, state: '', county: '' } : f));
  }, [series]);
  useEffect(() => {
    if (!series || !form.state) { setCounties([]); return; }
    fetch(`${API_URL}/counties/${form.state}?series=${encodeURIComponent(series.series)}`)
      .then(r => r.json()).then(d => setCounties(d.counties || [])).catch(() => setCounties([]));
  }, [series, form.state]);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v, ...(k === 'commodity' ? { series: '', state: '', county: '' } : {}), ...(k === 'state' ? { county: '' } : {}) }));

  async function submit(e) {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...overrides, series: series.series, state: form.state, county: form.county, year: Number(form.year), model: form.model }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      setResult(data);
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
        <h1>🌱 Crop Yield Predictor</h1>
        <p>Pick a crop, a county and a year</p>
      </header>
      <div className="main-content">
        <form onSubmit={submit} className="prediction-form">
          <div className="form-section">
            <h3>Crop and place</h3>
            <div className="input-group">
              <label>Crop</label>
              <select value={form.commodity} onChange={e => set('commodity', e.target.value)}>
                {meta.crops.map(c => <option key={c.commodity} value={c.commodity}>{c.commodity}</option>)}
              </select>
            </div>
            {crop?.series.length > 1 && (
              <div className="input-group">
                <label>Variety</label>
                <select value={series?.series} onChange={e => set('series', e.target.value)}>
                  {crop.series.map(s => <option key={s.series} value={s.series}>{label(s) || 'all classes'} ({s.unit.toLowerCase()})</option>)}
                </select>
              </div>
            )}
            <div className="input-group">
              <label>State</label>
              <select name="State" value={form.state} onChange={e => set('state', e.target.value)} required>
                <option value="">Select a state</option>
                {states.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <small style={{ color: '#666' }}>States with recent yield records for this variety</small>
            </div>
            <div className="input-group">
              <label>County</label>
              <select name="County" value={form.county} onChange={e => set('county', e.target.value)} required disabled={!counties.length}>
                <option value="">{form.state ? 'Select a county' : 'Choose a state first'}</option>
                {counties.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="input-group">
              <label>Year</label>
              <input type="number" name="Year" min="1981" max="2100" value={form.year} onChange={e => set('year', e.target.value)} required />
            </div>
          </div>

          <div className="form-section">
            <h3>Model</h3>
            <div className="input-group">
              <select value={form.model} onChange={e => set('model', e.target.value)}>
                {meta.models.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          </div>

          <details className="form-section">
            <summary><strong>Optional: override the county's normal weather and soil</strong></summary>
            {OVERRIDES.map(([title, fields]) => (
              <div key={title}>
                <h3>{title}</h3>
                <div className="input-grid">
                  {fields.map(([key, text]) => (
                    <div className="input-group" key={key}>
                      <label>{text}</label>
                      <input type="number" step="any" placeholder="County normal" value={overrides[key] ?? ''}
                             onChange={e => setOverrides(o => ({ ...o, [key]: e.target.value }))} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </details>

          <button type="submit" className="submit-btn" disabled={loading || !form.state || !form.county}>
            {loading ? 'Predicting…' : 'Predict yield'}
          </button>
        </form>

        {error && <div className="error-msg">{error}</div>}

        {result && (
          <div className="result-card">
            <h2>Predicted yield</h2>
            <p className="prediction-value">{fmt(result.prediction)} <span className="unit">{result.unit.toLowerCase()}</span></p>
            <p className="model-info">{result.commodity} · {form.county}, {form.state} · {form.year} · {result.model}</p>
            <p>Last observed here: <strong>{fmt(result.last_observed.yield)}</strong> in {result.last_observed.year}.
               Range across counties in {result.typical_range_years?.[0]}–{result.typical_range_years?.[1]}: {fmt(result.typical_range[0])}–{fmt(result.typical_range[1])}.</p>
            {result.expected_error_pct != null && (
              <p className="typical-range">
                On years the model never saw ({result.evaluated_on}), its median error for {result.commodity.toLowerCase()} was about {result.expected_error_pct}%.
              </p>
            )}
            <p className="typical-range">Weather and soil: {result.inputs}.
              {result.history === 'partial' && ' This year is not after the last observation, so the county history it uses is limited.'}</p>
          </div>
        )}
      </div>
    </div>
  );
}
