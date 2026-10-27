import React, { useState, useEffect } from 'react';
import { API_URL } from '../api';
import './../App.css';

function App() {
  const [metadata, setMetadata] = useState({ crops: [], states: [], available_models: [] });
  const [counties, setCounties] = useState([]);
  const [validStates, setValidStates] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const [formData, setFormData] = useState({
    Crop: '',
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

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch Metadata on Load
  useEffect(() => {
    fetch(`${API_URL}/metadata`)
      .then(res => res.json())
      .then(data => {
        setMetadata(data);
        setSelectedModel(data.current_model);
      })
      .catch(err => console.error("Failed to load metadata", err));
  }, []);

  // Fetch valid states when crop changes
  useEffect(() => {
    if (formData.Crop) {
      fetch(`${API_URL}/valid-states/${formData.Crop}`)
        .then(res => res.json())
        .then(data => setValidStates(data.states))
        .catch(err => {
          console.error("Failed to load valid states", err);
          setValidStates(metadata.states); // Fallback to all states
        });
      // Reset state and county when crop changes
      setFormData(prev => ({ ...prev, State: '', County: '' }));
    } else {
      setValidStates([]);
    }
  }, [formData.Crop, metadata.states]);

  // Fetch Counties when State changes
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
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);

    // Optionally switch model on backend
    fetch(`${API_URL}/models/${newModel}`, {
      method: 'POST'
    })
      .then(res => res.json())
      .then(data => console.log(data.message))
      .catch(err => console.error("Failed to switch model", err));
  };

  const validateInputs = () => {
    const errors = [];

    // Only validate if values are provided
    if (formData.tavg_growing_avg && (parseFloat(formData.tavg_growing_avg) < -20 || parseFloat(formData.tavg_growing_avg) > 45)) {
      errors.push("Average temperature should be -20°C to 45°C");
    }
    if (formData.tmin_growing_avg && (parseFloat(formData.tmin_growing_avg) < -30 || parseFloat(formData.tmin_growing_avg) > 40)) {
      errors.push("Minimum temperature should be -30°C to 40°C");
    }
    if (formData.tmax_growing_avg && (parseFloat(formData.tmax_growing_avg) < -10 || parseFloat(formData.tmax_growing_avg) > 50)) {
      errors.push("Maximum temperature should be -10°C to 50°C");
    }

    // Temperature relationships (if all provided)
    if (formData.tmin_growing_avg && formData.tavg_growing_avg && formData.tmax_growing_avg) {
      const tmin = parseFloat(formData.tmin_growing_avg);
      const tavg = parseFloat(formData.tavg_growing_avg);
      const tmax = parseFloat(formData.tmax_growing_avg);

      if (tmin > tavg) errors.push("Min temp cannot exceed avg temp");
      if (tmax < tavg) errors.push("Max temp cannot be less than avg temp");
      if (tmin > tmax) errors.push("Min temp cannot exceed max temp");
    }

    if (formData.prcp_growing_total) {
      const rainfall = parseFloat(formData.prcp_growing_total);
      if (rainfall < 0 || rainfall > 2000) {
        errors.push("Rainfall should be 0-2000mm");
      }
    }

    if (formData.rh_growing_avg) {
      const humidity = parseFloat(formData.rh_growing_avg);
      if (humidity < 0 || humidity > 100) {
        errors.push("Humidity must be 0-100%");
      }
    }

    if (formData.gdd_total) {
      const gdd = parseFloat(formData.gdd_total);
      if (gdd < 0 || gdd > 6000) {
        errors.push("GDD should be 0-6000");
      }
    }

    if (formData.heat_stress_days) {
      const heatDays = parseFloat(formData.heat_stress_days);
      if (heatDays < 0 || heatDays > 180) {
        errors.push("Heat stress days should be 0-180");
      }
    }

    // Soil validation (if all three provided)
    if (formData.clay_pct && formData.sand_pct && formData.silt_pct) {
      const clay = parseFloat(formData.clay_pct);
      const sand = parseFloat(formData.sand_pct);
      const silt = parseFloat(formData.silt_pct);
      const sum = clay + sand + silt;

      if (clay < 0 || clay > 100) errors.push("Clay % must be 0-100");
      if (sand < 0 || sand > 100) errors.push("Sand % must be 0-100");
      if (silt < 0 || silt > 100) errors.push("Silt % must be 0-100");

      if (sum < 95 || sum > 105) {
        errors.push(`Soil components should sum to ~100% (currently ${sum.toFixed(1)}%)`);
      }
    }

    if (formData.ph) {
      const ph = parseFloat(formData.ph);
      if (ph < 4 || ph > 9) {
        errors.push("pH should be 4-9");
      }
    }

    if (formData.Latitude) {
      const lat = parseFloat(formData.Latitude);
      if (lat < 24 || lat > 50) {
        errors.push("Latitude outside US range (24-50°N)");
      }
    }

    if (formData.Longitude) {
      const lon = parseFloat(formData.Longitude);
      if (lon < -125 || lon > -65) {
        errors.push("Longitude outside US range (-125 to -65°W)");
      }
    }

    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const validationErrors = validateInputs();
    if (validationErrors.length > 0) {
      setError(validationErrors.join("; "));
      setLoading(false);
      return;
    }

    try {
      const requestData = { ...formData, model: selectedModel };

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);  // Store entire response object
        if (data.warnings) {
          
        }
      } else {
        setError(data.error || "Prediction failed");
        if (data.valid_states) {
          setError(prev => prev + `. Valid states: ${data.valid_states.join(', ')}`);
        }
      }
    } catch (err) {
      setError("Network error. Ensure Flask backend is running.");
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <header>
        <h1>🌱 Crop Yield Predictor</h1>
        <p>ML-powered yield prediction with multiple models</p>
      </header>

      <div className="main-content">
        <form onSubmit={handleSubmit} className="prediction-form">

          {/* Model Selection */}
          <div className="form-section">
            <h3>Model Selection</h3>
            <div className="input-group">
              <label>ML Model</label>
              <select value={selectedModel} onChange={handleModelChange}>
                {metadata.available_models && metadata.available_models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
              <small style={{ color: '#666' }}>Different models may produce slightly different predictions</small>
            </div>
          </div>

          {/* Required Fields */}
          <div className="form-section">
            <h3>Location & Crop (Required)</h3>

            <div className="input-group">
              <label>Crop Type</label>
              <select name="Crop" value={formData.Crop} onChange={handleChange} required>
                <option value="">-- Select Crop --</option>
                {metadata.crops.map(crop => (
                  <option key={crop} value={crop}>{crop}</option>
                ))}
              </select>
            </div>

            <div className="input-group">
              <label>State</label>
              <select name="State" value={formData.State} onChange={handleChange} required>
                <option value="">-- Select State --</option>
                {(validStates.length > 0 ? validStates : metadata.states).map(state => (
                  <option key={state} value={state}>{state}</option>
                ))}
              </select>
              {validStates.length > 0 && formData.Crop && (
                <small style={{ color: '#666' }}>
                  Showing states where {formData.Crop} is typically grown
                </small>
              )}
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
              {!formData.State && <small style={{ color: '#999' }}>Select a state first</small>}
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
            {loading ? 'Calculating...' : 'Predict Yield'}
          </button>
        </form>

        {/* Results Display */}
        {result && (
          <div className="result-card">
            <h2>Predicted Yield</h2>
            <div className="prediction-value">
              {result.prediction.toFixed(2)} <span className="unit">{result.units}</span>
            </div>
            <p className="model-info">Using: {selectedModel}</p>
            {result.confidence && (
              <p className="confidence-info" style={{ marginTop: '5px' }}>
                Confidence: <strong style={{ color: result.confidence === 'High' ? 'green' : result.confidence === 'Medium' ? 'orange' : 'red' }}>
                  {result.confidence}
                </strong>
              </p>
            )}
            {result.typical_range && (
              <p className="typical-range" style={{ fontSize: '0.9em', color: '#666' }}>
                Typical range: {result.typical_range}
              </p>
            )}
            {result.defaults_source === 'county' && (
              <p style={{ fontSize: '0.85em', color: '#666' }}>
                Weather and soil fields left blank use this county's 1981–2018 averages.
              </p>
            )}
            {result.warnings && result.warnings.length > 0 && (
              <div className="warnings">
                {result.warnings.map((warning, idx) => (
                  <div key={idx} className="warning-msg">⚠️ {warning}</div>
                ))}
              </div>
            )}
          </div>
        )}

        {error && <div className="error-msg">{error}</div>}
      </div>
    </div>
  );
}

export default App;