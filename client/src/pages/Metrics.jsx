import React, { useEffect, useState } from 'react';
import { API_URL } from '../api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './Metrics.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function Metrics() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [featureEngineering, setFeatureEngineering] = useState(null);
  const [showFeatureEng, setShowFeatureEng] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  const [errorAnalysis, setErrorAnalysis] = useState(null);
  const [loadingErrorAnalysis, setLoadingErrorAnalysis] = useState(false);
  const [showErrorAnalysis, setShowErrorAnalysis] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/metrics`)
      .then(res => res.json())
      .then(data => {
        if (data.error) setError(data.error);
        else setData(data);
      })
      .catch(err => setError("Failed to fetch metrics"))
      .finally(() => setLoading(false));
  }, []);

  const loadFeatureEngineering = () => {
    if (featureEngineering && !featureEngineering.error) {
      setShowFeatureEng(!showFeatureEng);
      return;
    }
    setShowFeatureEng(true);
    fetch(`${API_URL}/feature-engineering`)
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        setFeatureEngineering(data);
      })
      .catch(err => {
        console.error("Failed to fetch feature engineering info", err);
        setFeatureEngineering({error: `Failed to load: ${err.message}`});
      });
  };

  const loadFeatureImportance = (modelName) => {
    setSelectedModel(modelName);
    setLoadingFeatures(true);
    const modelKey = modelName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_');
    fetch(`${API_URL}/feature-importance/${modelKey}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setFeatureImportance(null);
        } else {
          setFeatureImportance(data);
        }
      })
      .catch(err => {
        setFeatureImportance(null);
      })
      .finally(() => setLoadingFeatures(false));
  };

  const loadErrorAnalysis = (modelName) => {
    if (errorAnalysis && errorAnalysis.model_name === modelName && !showErrorAnalysis) {
      setShowErrorAnalysis(true);
      return;
    }
    setShowErrorAnalysis(true);
    setLoadingErrorAnalysis(true);
    const modelKey = modelName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_');
    fetch(`${API_URL}/error-analysis/${modelKey}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setErrorAnalysis(null);
        } else {
          setErrorAnalysis(data);
        }
      })
      .catch(err => {
        setErrorAnalysis(null);
      })
      .finally(() => setLoadingErrorAnalysis(false));
  };

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;

  // Color mapping by model type
  const colorMap = {
    'Baseline': '#9e9e9e',
    'Classical ML': '#4fc3f7',
    'Ensemble/Advanced': '#43a047'
  };

  // Prepare data for charts (exclude baselines from main charts)
  // const mlModels = data.filter(d => d.Type !== 'Baseline');
  const mlModels = data;
  const allModels = [...data];

  // Sort for R² chart (descending)
  const r2Data = [...mlModels].sort((a, b) => b['R²'] - a['R²']);

  // Sort for training time (ascending)
  const timeData = [...mlModels].sort((a, b) => a['Training Time (s)'] - b['Training Time (s)']);

  // Sort for MAE (ascending - lower is better)
  const maeData = [...mlModels].sort((a, b) => a['MAE'] - b['MAE']);

  // Sort for RMSE (ascending - lower is better)
  const rmseData = [...mlModels].sort((a, b) => a['RMSE'] - b['RMSE']);

  // Chart 1: R² Score Bar Chart
  const r2ChartData = {
    labels: r2Data.map(d => d.Model),
    datasets: [{
      label: 'R² Score',
      data: r2Data.map(d => d['R²']),
      backgroundColor: r2Data.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const r2Options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: { display: true, text: 'R² Score' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 2: Training Time Bar Chart
  const timeChartData = {
    labels: timeData.map(d => d.Model),
    datasets: [{
      label: 'Training Time (s)',
      data: timeData.map(d => d['Training Time (s)']),
      backgroundColor: timeData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const timeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Training Time (seconds)' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 3: MAE Bar Chart
  const maeChartData = {
    labels: maeData.map(d => d.Model),
    datasets: [{
      label: 'MAE',
      data: maeData.map(d => d['MAE']),
      backgroundColor: maeData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const maeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Mean Absolute Error' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  // Chart 4: RMSE Bar Chart
  const rmseChartData = {
    labels: rmseData.map(d => d.Model),
    datasets: [{
      label: 'RMSE',
      data: rmseData.map(d => d['RMSE']),
      backgroundColor: rmseData.map(d => colorMap[d.Type]),
      borderRadius: 8,
    }]
  };

  const rmseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Root Mean Square Error' }
      },
      x: {
        ticks: {
          autoSkip: false,
          maxRotation: 45,
          minRotation: 45,
          font: { size: 10 }
        }
      }
    }
  };

  return (
    <div className="metrics-container">
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '10px'}}>
        <div>
          <h2>Model Performance Comparison</h2>
          <p className="metrics-subtitle">Comparing {data.length} models across multiple performance metrics</p>
        </div>
        <button 
          onClick={loadFeatureEngineering}
          style={{
            padding: '12px 24px', 
            cursor: 'pointer',
            backgroundColor: showFeatureEng ? '#f44336' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: 'bold',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            transition: 'all 0.3s ease',
            whiteSpace: 'nowrap'
          }}
        >
          {showFeatureEng ? '🔽 Hide' : '🔼 Show'} Feature Engineering Info
        </button>
      </div>

      {showFeatureEng && (
        <div style={{
          marginBottom: '30px',
          padding: '20px',
          backgroundColor: '#f5f5f5',
          borderRadius: '8px',
          border: '1px solid #ddd'
        }}>
          {!featureEngineering ? (
            <p>Loading feature engineering info...</p>
          ) : featureEngineering.error ? (
            <p style={{color: 'red'}}>Error: {featureEngineering.error}</p>
          ) : (
            <>
              <h3>Feature Engineering ({featureEngineering.total_engineered_features} Features)</h3>
              <p style={{color: '#666', marginBottom: '20px'}}>
                {featureEngineering.description}
              </p>
              
              {Object.entries(featureEngineering.categories).map(([category, features]) => (
                <div key={category} style={{marginBottom: '25px'}}>
                  <h4 style={{color: '#333', marginBottom: '10px', textTransform: 'capitalize'}}>
                    {category.replace(/_/g, ' ')} ({Object.keys(features).length} features)
                  </h4>
                  <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '15px'}}>
                    {Object.entries(features).map(([featName, featInfo]) => (
                      <div key={featName} style={{
                        padding: '12px',
                        backgroundColor: 'white',
                        borderRadius: '5px',
                        border: '1px solid #ddd'
                      }}>
                        <strong style={{color: '#2196F3'}}>{featName}</strong>
                        <div style={{marginTop: '8px', fontSize: '0.9em'}}>
                          <div style={{color: '#666', marginBottom: '5px'}}>
                            <strong>Formula:</strong> {featInfo.formula}
                          </div>
                          <div style={{color: '#555'}}>
                            {featInfo.description}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Charts Section */}
      <div className="charts-grid">
        {/* Chart 1: R² Score Comparison */}
        <div className="chart-card">
          <h3>Model Accuracy (R² Score)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={r2ChartData} options={r2Options} />
          </div>
        </div>

        {/* Chart 2: Training Time Comparison */}
        <div className="chart-card">
          <h3>Training Time (seconds)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={timeChartData} options={timeOptions} />
          </div>
        </div>

        {/* Chart 3: MAE Comparison */}
        <div className="chart-card">
          <h3>Mean Absolute Error (MAE)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={maeChartData} options={maeOptions} />
          </div>
        </div>

        {/* Chart 4: RMSE Comparison */}
        <div className="chart-card">
          <h3>Root Mean Square Error (RMSE)</h3>
          <div style={{ height: '300px' }}>
            <Bar data={rmseChartData} options={rmseOptions} />
          </div>
        </div>
      </div>

      {/* Detailed Table */}
      <div className="table-section">
        <h3>Detailed Metrics Table</h3>
        <div className="table-container">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>MAPE (%)</th>
                <th>Within 10%</th>
                <th>Training Time (s)</th>
                <th>Complexity</th>
                <th>Interpretability</th>
                <th>Type</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {allModels.map((row, idx) => (
                <tr key={idx} className={`row - type - ${row.Type.replace(/\s+/g, '-').toLowerCase()} `}>
                  <td><strong>{row.Model}</strong></td>
                  <td>{row["R²"].toFixed(3)}</td>
                  <td>{row.RMSE.toFixed(2)}</td>
                  <td>{row.MAE.toFixed(2)}</td>
                  <td>{row["MAPE (%)"].toFixed(2)}</td>
                  <td>{row["Within 10%"].toFixed(2)}</td>
                  <td>{row["Training Time (s)"].toFixed(2)}</td>
                  <td>{row.Complexity}</td>
                  <td>{row.Interpretability}</td>
                  <td><span className={`badge badge - ${row.Type.replace(/\s+/g, '-').toLowerCase()} `}>{row.Type}</span></td>
                  <td>
                    <div style={{display: 'flex', gap: '5px', flexWrap: 'wrap'}}>
                      <button 
                        onClick={() => loadFeatureImportance(row.Model)}
                        style={{
                          padding: '6px 12px', 
                          cursor: 'pointer',
                          backgroundColor: '#2196F3',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}
                      >
                        Features
                      </button>
                      <button 
                        onClick={() => loadErrorAnalysis(row.Model)}
                        style={{
                          padding: '6px 12px', 
                          cursor: 'pointer',
                          backgroundColor: '#FF9800',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}
                      >
                        Errors
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance Section */}
      {selectedModel && (
        <div className="table-section" style={{marginTop: '30px'}}>
          <h3>Feature Importance: {selectedModel}</h3>
          {loadingFeatures ? (
            <p>Loading feature importance...</p>
          ) : featureImportance ? (
            <div className="table-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {featureImportance.top_10.map((feat, idx) => (
                    <tr key={idx}>
                      <td>{idx + 1}</td>
                      <td><strong>{feat.Feature}</strong></td>
                      <td>{feat.Importance.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{color: '#666', fontStyle: 'italic'}}>
              Feature importance not available for this model. Only tree-based models (Random Forest, XGBoost, LightGBM, Gradient Boosting) provide feature importance.
            </p>
          )}
        </div>
      )}

      {/* Error Analysis Section */}
      {showErrorAnalysis && (
        <div className="table-section" style={{marginTop: '30px'}}>
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px'}}>
            <h3>Error Analysis: {errorAnalysis?.model_name || selectedModel}</h3>
            <button 
              onClick={() => setShowErrorAnalysis(false)}
              style={{
                padding: '5px 15px',
                cursor: 'pointer',
                backgroundColor: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '12px'
              }}
            >
              Close
            </button>
          </div>
          
          {loadingErrorAnalysis ? (
            <p>Loading error analysis...</p>
          ) : errorAnalysis ? (
            <>
              {/* Overall Statistics */}
              <div style={{marginBottom: '25px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px'}}>
                <h4 style={{marginTop: 0}}>Overall Error Statistics</h4>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px'}}>
                  <div>
                    <strong>Mean Absolute Error:</strong> {errorAnalysis.overall_statistics.mean_absolute_error.toFixed(2)}
                  </div>
                  <div>
                    <strong>Mean Absolute % Error:</strong> {errorAnalysis.overall_statistics.mean_absolute_percent_error.toFixed(2)}%
                  </div>
                  <div>
                    <strong>Max Absolute % Error:</strong> {errorAnalysis.overall_statistics.max_absolute_percent_error.toFixed(2)}%
                  </div>
                  <div>
                    <strong>Std of % Errors:</strong> {errorAnalysis.overall_statistics.std_percent_errors.toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Top 10 Errors */}
              <div style={{marginBottom: '25px'}}>
                <h4>Top 10 Largest Percentage Errors</h4>
                <div className="table-container">
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>True Value</th>
                        <th>Predicted Value</th>
                        <th>Error</th>
                        <th>% Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {errorAnalysis.top_10_errors.map((err, idx) => (
                        <tr key={idx}>
                          <td>{err.rank}</td>
                          <td>{err.true_value.toFixed(2)}</td>
                          <td>{err.predicted_value.toFixed(2)}</td>
                          <td style={{color: err.error > 0 ? '#f44336' : '#4CAF50'}}>
                            {err.error > 0 ? '+' : ''}{err.error.toFixed(2)}
                          </td>
                          <td style={{color: err.percent_error > 100 ? '#f44336' : '#FF9800'}}>
                            {err.percent_error.toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Error Distribution */}
              <div>
                <h4>Error Distribution by Percent Error Range</h4>
                <div className="table-container">
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Error Range</th>
                        <th>Count</th>
                        <th>Percentage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {errorAnalysis.error_distribution.map((dist, idx) => (
                        <tr key={idx}>
                          <td><strong>{dist.range}</strong></td>
                          <td>{dist.count.toLocaleString()}</td>
                          <td>{dist.percentage.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <p style={{color: '#666', fontStyle: 'italic'}}>
              Error analysis not available for this model. Run training to generate error analysis.
            </p>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="legend-section">
        <h4>Model Type Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Baseline'] }}></span>
            <span>Baseline Models</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Classical ML'] }}></span>
            <span>Classical ML</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: colorMap['Ensemble/Advanced'] }}></span>
            <span>Ensemble/Advanced</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Metrics;
