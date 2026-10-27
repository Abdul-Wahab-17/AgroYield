import { useState, useEffect } from 'react';
import { API_URL } from '../api';
import './../App.css';

function Regional() {
  const [metadata, setMetadata] = useState({ crops: [], states: [], available_models: [] });
  const [counties, setCounties] = useState([]);
  const [validStates, setValidStates] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const [formData, setFormData] = useState({
    State: '',
    County: '',
    Year: new Date().getFullYear(),
    prcp_growing_total: '',
    tavg_growing_avg: '',
    tmin_growing_avg: '',
    tmax_growing_avg: '',
    rh_growing_avg: '',
    gdd_total: '',
    heat_stress_days: '',
    temp_range_avg: '',
    clay_pct: '',
    sand_pct: '',
    silt_pct: '',
    organic_matter_pct: '',
    ph: '',
    bulk_density: '',
    cec: '',
    awc: '',
    Latitude: '',
    Longitude: ''
  });

  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load metadata
  useEffect(() => {
    fetch(`${API_URL}/metadata`)
      .then(res => res.json())
      .then(data => {
        setMetadata(data);
        setSelectedModel(data.current_model);
      })
      .catch(err => console.error("Failed to load metadata", err));
  }, []);

useEffect(() => {
  if (formData.State && !metadata.states.includes(formData.State)) {
    setFormData(prev => ({ ...prev, State: '' }));
  }
}, [formData.State, metadata.states]);

  // Fetch valid states when crop changes — here crop is iterated, so we just consider all crops
  useEffect(() => {
    if (metadata.states.length > 0) {
      setValidStates(metadata.states);
    }
  }, [metadata.states]);

  // Fetch counties when state changes
  useEffect(() => {
    if (formData.State) {
      fetch(`${API_URL}/counties/${formData.State}`)
        .then(res => res.json())
        .then(data => setCounties(data.counties))
        .catch(err => console.error("Failed to load counties", err));
      setFormData(prev => ({ ...prev, County: '' }));
    } else {
      setCounties([]);
    }
  }, [formData.State]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    fetch(`${API_URL}/models/${newModel}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => console.log(data.message))
      .catch(err => console.error("Failed to switch model", err));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults([]);

    if (!formData.State || !formData.County) {
      setError("State and County are required");
      setLoading(false);
      return;
    }

    try {
      const cropList = metadata.crops;
      const predictions = [];

      for (const crop of cropList) {
        const requestData = { ...formData, Crop: crop, model: selectedModel };
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        });
        const data = await response.json();
        if (response.ok) {
          predictions.push({ crop, ...data });
        } else if (data.valid_states) {
          // The API refuses crops this state has no yield records for
          predictions.push({ crop, notGrown: true });
        } else {
          predictions.push({ crop, error: data.error || "Prediction failed" });
        }
      }

      setResults(predictions);
    } catch (err) {
      setError("Network error. Ensure Flask backend is running.");
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <header>
        <h1>🌱 Regional Crop Yield Prediction</h1>
        <p>Enter State and County to get predictions for all crops</p>
      </header>

      <div className="main-content">
        <form onSubmit={handleSubmit} className="prediction-form">

          {/* Model Selection */}
          <div className="form-section">
            <h3>Model Selection</h3>
            <div className="input-group">
              <label>ML Model</label>
              <select value={selectedModel} onChange={handleModelChange}>
                {metadata.available_models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Required Fields */}
          <div className="form-section">
            <h3>Location (Required)</h3>
            <div className="input-group">
              <label>State</label>
              <select name="State" value={formData.State} onChange={handleChange} required>
                <option value="">-- Select State --</option>
                {validStates.map(state => (
                  <option key={state} value={state}>{state}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>County</label>
              <select
                name="County"
                value={formData.County}
                onChange={handleChange}
                required
                disabled={!formData.State}
              >
                <option value="">-- Select County --</option>
                {counties.map(county => (
                  <option key={county} value={county}>{county}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Year Field */}
          <div className="form-section">
            <h3>Year</h3>
            <div className="input-group">
              <label>Year</label>
              <input
                type="number"
                name="Year"
                min="1980"
                max="2050"
                value={formData.Year}
                onChange={handleChange}
              />
            </div>
          </div>

          {/* Optional Weather Fields */}
          <div className="form-section">
            <h3>Weather Conditions (Optional)</h3>
            <div className="input-grid">
              <div className="input-group">
                <label>Total Rainfall (mm)</label>
                <input type="number" name="prcp_growing_total" placeholder="County average" value={formData.prcp_growing_total} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Avg Temperature (°C)</label>
                <input type="number" name="tavg_growing_avg" placeholder="County average" value={formData.tavg_growing_avg} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Min Temperature (°C)</label>
                <input type="number" name="tmin_growing_avg" placeholder="County average" value={formData.tmin_growing_avg} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Max Temperature (°C)</label>
                <input type="number" name="tmax_growing_avg" placeholder="County average" value={formData.tmax_growing_avg} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Relative Humidity (%)</label>
                <input type="number" name="rh_growing_avg" placeholder="County average" value={formData.rh_growing_avg} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Growing Degree Days</label>
                <input type="number" name="gdd_total" placeholder="County average" value={formData.gdd_total} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Heat Stress Days</label>
                <input type="number" name="heat_stress_days" placeholder="County average" value={formData.heat_stress_days} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Temperature Range (°C)</label>
                <input type="number" name="temp_range_avg" placeholder="County average" value={formData.temp_range_avg} onChange={handleChange} />
              </div>
            </div>
          </div>

          {/* Optional Soil Fields */}
          <div className="form-section">
            <h3>Soil Conditions (Optional)</h3>
            <div className="input-grid">
              <div className="input-group">
                <label>Clay (%)</label>
                <input type="number" name="clay_pct" placeholder="County average" value={formData.clay_pct} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Sand (%)</label>
                <input type="number" name="sand_pct" placeholder="County average" value={formData.sand_pct} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Silt (%)</label>
                <input type="number" name="silt_pct" placeholder="County average" value={formData.silt_pct} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Organic Matter (%)</label>
                <input type="number" name="organic_matter_pct" placeholder="County average" value={formData.organic_matter_pct} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>pH</label>
                <input type="number" name="ph" placeholder="County average" step="0.1" value={formData.ph} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Bulk Density (g/cm³)</label>
                <input type="number" name="bulk_density" placeholder="County average" step="0.1" value={formData.bulk_density} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>CEC</label>
                <input type="number" name="cec" placeholder="County average" value={formData.cec} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>AWC</label>
                <input type="number" name="awc" placeholder="County average" step="0.01" value={formData.awc} onChange={handleChange} />
              </div>
            </div>
          </div>

          {/* Optional Geographic Fields */}
          <div className="form-section">
            <h3>Geographic Location (Optional)</h3>
            <div className="input-grid">
              <div className="input-group">
                <label>Latitude</label>
                <input type="number" name="Latitude" placeholder="County average" step="0.01" value={formData.Latitude} onChange={handleChange} />
              </div>
              <div className="input-group">
                <label>Longitude</label>
                <input type="number" name="Longitude" placeholder="County average" step="0.01" value={formData.Longitude} onChange={handleChange} />
              </div>
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Calculating...' : 'Predict All Crops'}
          </button>
        </form>

        {/* Results */}
        {results.length > 0 && (
          <div className="result-card">
            <h2>Predictions</h2>
            {results.filter(res => !res.notGrown).map((res, idx) => (
              <div key={idx} className="prediction-item">
                <strong>{res.crop}</strong>: {res.error ? res.error : (
                  <span>
                    {res.prediction.toFixed(2)} {res.units}
                    {res.confidence && (
                      <span style={{ marginLeft: '10px', fontSize: '0.85em', color: res.confidence === 'High' ? 'green' : res.confidence === 'Medium' ? 'orange' : 'red' }}>
                        (Confidence: {res.confidence})
                      </span>
                    )}
                  </span>
                )}
                {res.warnings && res.warnings.length > 0 && (
                  <div className="warnings">
                    {res.warnings.map((w, i) => <div key={i}>⚠️ {w}</div>)}
                  </div>
                )}
              </div>
            ))}
            {results.some(res => res.notGrown) && (
              <p style={{ marginTop: '12px', fontSize: '0.9em', color: '#666' }}>
                Not grown in {formData.State} (no yield records): {results.filter(res => res.notGrown).map(res => res.crop).join(', ')}
              </p>
            )}
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
      </div>
    </div>
  );
}

export default Regional;
