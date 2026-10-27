from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from fetch_artifacts import ensure_artifacts

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Use __file__ to get the directory where app.py is located, regardless of working directory
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = Path(os.environ.get("AGROYIELD_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR

# Model file -> display name. Names match the "Model" column of model_comparison.csv,
# so per-model R² (confidence), feature importance and error analysis line up.
MODEL_NAMES = {
    'random_forest.pkl': 'Random Forest',
    'xgboost.pkl': 'XGBoost',
    'lightgbm.pkl': 'LightGBM',
    'catboost.pkl': 'CatBoost',
    'gradient_boosting.pkl': 'Gradient Boosting',
    'voting_ensemble.pkl': 'Voting Ensemble',
    'stacking_ensemble.pkl': 'Stacking Ensemble',
    'decision_tree.pkl': 'Decision Tree',
    'k-nn.pkl': 'K-NN',
    'linear_regression.pkl': 'Linear Regression',
    'ridge_regression.pkl': 'Ridge Regression',
    'svr.pkl': 'SVR',
}

ensure_artifacts(ARTIFACTS_DIR)

# --- GLOBAL VARIABLES ---
crop_encoder = None
state_encoder = None
county_encoder = None
numerical_scaler = None
weather_scaler = None
soil_scaler = None
geo_scaler = None
AVAILABLE_MODELS = {}
CURRENT_MODEL = None
FEATURE_ORDER = None
PREPROCESSING_METADATA = None
VALID_COMBINATIONS = None
DEFAULTS = None
COUNTY_DEFAULTS = {}
MODEL_PERFORMANCE = {}

# CRITICAL: Define crop-specific units based on training data analysis
CROP_UNITS = {
    'CORN': {'unit': 'Bu/Acre', 'typical_range': (50, 200)},
    'SOYBEANS': {'unit': 'Bu/Acre', 'typical_range': (20, 70)},
    'WHEAT': {'unit': 'Bu/Acre', 'typical_range': (30, 100)},
    'OATS': {'unit': 'Bu/Acre', 'typical_range': (30, 100)},
    'BARLEY': {'unit': 'Bu/Acre', 'typical_range': (30, 100)},
    'SORGHUM': {'unit': 'Bu/Acre', 'typical_range': (30, 100)},
    'COTTON': {'unit': 'lbs/acre', 'typical_range': (400, 1200)},
    'BEANS': {'unit': 'lbs/acre', 'typical_range': (1000, 2500)},
    'PEANUTS': {'unit': 'lbs/acre', 'typical_range': (2000, 4000)},
    'RICE': {'unit': 'lbs/acre', 'typical_range': (4000, 8000)},
    'HAY': {'unit': 'tons/acre', 'typical_range': (1.5, 4.5)},
    'SUNFLOWER': {'unit': 'lbs/acre', 'typical_range': (800, 2000)},
}

# Feature Engineering Documentation
FEATURE_ENGINEERING_INFO = {
    'time_features': {
        'Year_Since_Start': {
            'formula': 'Year - 1980',
            'description': 'Years since dataset start (captures temporal trends)',
            'category': 'Temporal'
        },
        'Year_Squared': {
            'formula': 'Year²',
            'description': 'Non-linear time component (captures acceleration/deceleration)',
            'category': 'Temporal'
        }
    },
    'climate_interactions': {
        'temp_prcp_interaction': {
            'formula': 'tavg_growing_avg × prcp_growing_total',
            'description': 'Temperature-precipitation interaction (captures combined climate effects)',
            'category': 'Climate'
        },
        'gdd_per_day': {
            'formula': 'gdd_total / 180',
            'description': 'Growing degree days per day (normalized growing season intensity)',
            'category': 'Climate'
        },
        'moisture_stress': {
            'formula': 'prcp_growing_total / (tavg_growing_avg + 1)',
            'description': 'Moisture availability relative to temperature (drought indicator)',
            'category': 'Climate'
        },
        'temp_variability': {
            'formula': 'temp_range_avg × heat_stress_days',
            'description': 'Temperature variability combined with heat stress frequency',
            'category': 'Climate'
        }
    },
    'soil_features': {
        'soil_quality_score': {
            'formula': 'organic_matter_pct × 0.3 + (7 - |ph - 6.5|) × 0.2 + awc × 100 × 0.3 + cec × 0.2',
            'description': 'Composite soil quality index (weighted combination of key soil properties)',
            'category': 'Soil'
        },
        'loam_index': {
            'formula': '100 - (|clay_pct - 20| + |sand_pct - 40| + |silt_pct - 40|)',
            'description': 'Proximity to ideal loam texture (20% clay, 40% sand, 40% silt)',
            'category': 'Soil'
        }
    },
    'geographic_features': {
        'lat_squared': {
            'formula': 'Latitude²',
            'description': 'Non-linear latitude effect (captures climate zones)',
            'category': 'Geographic'
        },
        'long_squared': {
            'formula': 'Longitude²',
            'description': 'Non-linear longitude effect',
            'category': 'Geographic'
        },
        'lat_long_interaction': {
            'formula': 'Latitude × Longitude',
            'description': 'Geographic interaction (captures regional climate patterns)',
            'category': 'Geographic'
        }
    }
}

def apply_feature_engineering(df):
    """
    Apply feature engineering transformations to input dataframe.
    Creates 13 engineered features from raw inputs.
    
    Args:
        df: DataFrame with raw input features
        
    Returns:
        DataFrame with engineered features added
    """
    # Time-based features
    df['Year_Since_Start'] = df['Year'] - 1980
    df['Year_Squared'] = df['Year'] ** 2
    
    # Climate interaction features
    df['temp_prcp_interaction'] = df['tavg_growing_avg'] * df['prcp_growing_total']
    df['gdd_per_day'] = df['gdd_total'] / 180
    df['moisture_stress'] = df['prcp_growing_total'] / (df['tavg_growing_avg'] + 1)
    df['temp_variability'] = df['temp_range_avg'] * df['heat_stress_days']
    
    # Soil quality composite
    df['soil_quality_score'] = (
        df['organic_matter_pct'] * 0.3 +
        (7 - abs(df['ph'] - 6.5)) * 0.2 +
        df['awc'] * 100 * 0.3 +
        df['cec'] * 0.2
    )
    
    # Texture-based features
    df['loam_index'] = 100 - (abs(df['clay_pct'] - 20) + 
                               abs(df['sand_pct'] - 40) + 
                               abs(df['silt_pct'] - 40))
    
    # Geographic features
    df['lat_squared'] = df['Latitude'] ** 2
    df['long_squared'] = df['Longitude'] ** 2
    df['lat_long_interaction'] = df['Latitude'] * df['Longitude']
    
    return df

print("="*80)
print("LOADING ARTIFACTS")
print("="*80)

try:
    # Load Encoders
    print("\n1. Loading Encoders...")
    with open(ARTIFACTS_DIR / 'crop_encoder.pkl', 'rb') as f:
        crop_encoder = pickle.load(f)
    print(f"   ✓ Crop Encoder: {len(crop_encoder.classes_)} crops")
    
    with open(ARTIFACTS_DIR / 'state_encoder.pkl', 'rb') as f:
        state_encoder = pickle.load(f)
    print(f"   ✓ State Encoder: {len(state_encoder.classes_)} states")
    
    with open(ARTIFACTS_DIR / 'county_encoder.pkl', 'rb') as f:
        county_encoder = pickle.load(f)
    print(f"   ✓ County Encoder: {len(county_encoder.classes_)} counties")
    
    # Load Scalers
    print("\n2. Loading Scalers...")
    with open(ARTIFACTS_DIR / 'numerical_scaler.pkl', 'rb') as f:
        numerical_scaler = pickle.load(f)
    print("   ✓ Numerical Scaler")
    
    with open(ARTIFACTS_DIR / 'weather_scaler.pkl', 'rb') as f:
        weather_scaler = pickle.load(f)
    print("   ✓ Weather Scaler")
    
    with open(ARTIFACTS_DIR / 'soil_scaler.pkl', 'rb') as f:
        soil_scaler = pickle.load(f)
    print("   ✓ Soil Scaler")
    
    with open(ARTIFACTS_DIR / 'geo_scaler.pkl', 'rb') as f:
        geo_scaler = pickle.load(f)
    print("   ✓ Geo Scaler")
    
    # Load Feature Names
    print("\n3. Loading Feature Order...")
    with open(ARTIFACTS_DIR / 'feature_names.pkl', 'rb') as f:
        FEATURE_ORDER = pickle.load(f)
    print(f"   ✓ Feature Order: {len(FEATURE_ORDER)} features")
    
    # Load Preprocessing Metadata
    print("\n4. Loading Preprocessing Metadata...")
    with open(ARTIFACTS_DIR / 'preprocessing_metadata.json', 'r') as f:
        PREPROCESSING_METADATA = json.load(f)
    print(f"   ✓ Metadata Loaded")
    
    # Load Valid Combinations
    valid_combos_path = ARTIFACTS_DIR / 'valid_crop_state_combinations.json'
    if valid_combos_path.exists():
        with open(valid_combos_path, 'r') as f:
            VALID_COMBINATIONS = json.load(f)
        print(f"   ✓ Valid Crop-State Combinations")

    # Per-county location, soil and average weather (used for inputs left blank)
    county_defaults_path = ARTIFACTS_DIR / 'county_defaults.json'
    if county_defaults_path.exists():
        with open(county_defaults_path, 'r') as f:
            COUNTY_DEFAULTS = json.load(f)
        print(f"   ✓ County Defaults ({len(COUNTY_DEFAULTS)} counties)")

    # Load Model Performance (for Confidence Estimation)
    csv_path = ARTIFACTS_DIR / 'model_comparison.csv'
    if csv_path.exists():
        perf_df = pd.read_csv(csv_path)
        # Create dictionary: Model Name -> R2
        # Use simple normalization for matching
        for _, row in perf_df.iterrows():
             MODEL_PERFORMANCE[row['Model']] = float(row['R²'])
    print(f"   ✓ Model Performance Loaded ({len(MODEL_PERFORMANCE)} models)")
    
    # Load every known model that is present (a deployment may ship only a subset)
    print("\n5. Loading Models...")
    for model_file, model_name in MODEL_NAMES.items():
        model_path = MODELS_DIR / model_file
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                AVAILABLE_MODELS[model_name] = {
                    'model': model,
                    'file': model_file,
                    'type': type(model).__name__
                }
                print(f"   ✓ {model_name}: {type(model).__name__}")
            except Exception as e:
                print(f"   ✗ Failed to load {model_file}: {e}")

    if not AVAILABLE_MODELS:
        raise Exception("No models found!")

    # Default to the most accurate model that was loaded
    CURRENT_MODEL = max(AVAILABLE_MODELS, key=lambda name: MODEL_PERFORMANCE.get(name, float('-inf')))
    print(f"\n   Default Model: {CURRENT_MODEL}")
    
    # Set defaults
    DEFAULTS = {
        'tavg_growing_avg': 20.0, 'tmin_growing_avg': 14.0, 'tmax_growing_avg': 26.0,
        'prcp_growing_total': 500.0, 'rh_growing_avg': 65.0, 'gdd_total': 2500.0,
        'heat_stress_days': 5.0, 'temp_range_avg': 12.0,
        'clay_pct': 20.0, 'sand_pct': 40.0, 'silt_pct': 40.0, 'organic_matter_pct': 2.5,
        'ph': 6.5, 'bulk_density': 1.3, 'cec': 15.0, 'awc': 0.15,
        'Latitude': 38.0, 'Longitude': -95.0, 'Year': 2024
    }
    
    print("\n" + "="*80)
    print("✅ ALL ARTIFACTS LOADED")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    raise



def defaults_for(state, county):
    """Inputs to use when the request leaves them blank: the county's own
    averages if known, otherwise generic values."""
    county_values = COUNTY_DEFAULTS.get(f"{str(county).upper()}_{str(state).upper()}")
    if county_values:
        return {**DEFAULTS, **county_values}, 'county'
    return DEFAULTS, 'generic'


@app.route('/models', methods=['GET'])
def get_models():
    models_info = []
    for name, info in AVAILABLE_MODELS.items():
        models_info.append({
            'name': name,
            'type': info['type'],
            'is_current': name == CURRENT_MODEL
        })
    return jsonify({
        'models': models_info,
        'current_model': CURRENT_MODEL
    })

@app.route('/models/<model_name>', methods=['POST'])
def set_model(model_name):
    global CURRENT_MODEL
    
    for name in AVAILABLE_MODELS.keys():
        if name.lower() == model_name.lower():
            CURRENT_MODEL = name
            return jsonify({
                'message': f'Switched to {name}',
                'model': CURRENT_MODEL
            })
    
    return jsonify({'error': f'Model {model_name} not found'}), 404

@app.route('/metadata', methods=['GET'])
def get_metadata():
    try:
        return jsonify({
            'crops': crop_encoder.classes_.tolist(),
            'states': state_encoder.classes_.tolist(),
            'current_model': CURRENT_MODEL,
            'available_models': list(AVAILABLE_MODELS.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/counties/<state>', methods=['GET'])
def get_counties(state):
    try:
        state_upper = state.upper()
        counties = [
            c.replace(f'_{state_upper}', '')
            for c in county_encoder.classes_
            if c.endswith(f'_{state_upper}')
        ]
        return jsonify({'counties': sorted(counties)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/valid-states/<crop>', methods=['GET'])
def get_valid_states(crop):
    try:
        if VALID_COMBINATIONS and crop in VALID_COMBINATIONS:
            return jsonify({'states': VALID_COMBINATIONS[crop]})
        return jsonify({'states': state_encoder.classes_.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    print("\n" + "="*80)
    print("PREDICTION REQUEST")
    print("="*80)
    
    requested_model = data.get('model', CURRENT_MODEL)
    if requested_model not in AVAILABLE_MODELS:
        requested_model = CURRENT_MODEL
    
    model = AVAILABLE_MODELS[requested_model]['model']
    
    try:
        # Prepare input
        input_data = {}
        input_data['Crop'] = data.get('Crop')
        input_data['State'] = data.get('State')
        input_data['County'] = data.get('County', 'UNKNOWN').upper()
        
        if not input_data['Crop'] or not input_data['State']:
            return jsonify({'error': 'Crop and State required'}), 400
        
        # Validate crop-state combination
        if VALID_COMBINATIONS and input_data['Crop'] in VALID_COMBINATIONS:
            valid_states = VALID_COMBINATIONS[input_data['Crop']]
            if input_data['State'] not in valid_states:
                return jsonify({
                    'error': f"{input_data['Crop']} not typically grown in {input_data['State']}",
                    'valid_states': valid_states[:10]
                }), 400
        
        # Apply defaults
        defaults, defaults_source = defaults_for(input_data['State'], input_data['County'])
        for key, default_val in defaults.items():
            val = data.get(key)
            input_data[key] = float(val) if val not in [None, ""] else float(default_val)
        
        print(f"Model: {requested_model}")
        print(f"Crop: {input_data['Crop']}, State: {input_data['State']}")
        
        df = pd.DataFrame([input_data])
        
        # Feature Engineering (on RAW values)
        df = apply_feature_engineering(df)
        
        # Encoding
        crop_val = df['Crop'].iloc[0]
        if crop_val not in crop_encoder.classes_:
            return jsonify({'error': f'Unknown crop: {crop_val}'}), 400
        df['Crop_Encoded'] = crop_encoder.transform([crop_val])[0]
        
        state_val = df['State'].iloc[0]
        if state_val not in state_encoder.classes_:
            return jsonify({'error': f'Unknown state: {state_val}'}), 400
        df['State_Encoded'] = state_encoder.transform([state_val])[0]
        
        # County with fallback
        county_val = f"{df['County'].iloc[0]}_{state_val.upper()}"
        if county_val in county_encoder.classes_:
            df['County_Encoded'] = county_encoder.transform([county_val])[0]
        else:
            state_counties = [c for c in county_encoder.classes_ 
                            if c.endswith(f"_{state_val.upper()}")]
            if state_counties:
                df['County_Encoded'] = county_encoder.transform([state_counties[0]])[0]
            else:
                df['County_Encoded'] = 0
        
        # CRITICAL FIX: Scale features BEFORE calculating extreme_heat_flag
        weather_cols = [
            'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg',
            'prcp_growing_total', 'rh_growing_avg', 'gdd_total',
            'heat_stress_days', 'temp_range_avg',
            'temp_prcp_interaction', 'gdd_per_day', 'moisture_stress',
            'temp_variability'
        ]
        soil_cols = [
            'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct',
            'ph', 'bulk_density', 'cec', 'awc',
            'soil_quality_score', 'loam_index'
        ]
        geo_cols = ['Latitude', 'Longitude', 'lat_squared', 'long_squared', 'lat_long_interaction']
        time_cols = ['Year', 'Year_Since_Start', 'Year_Squared']
        
        # Scale features
        df[weather_cols] = weather_scaler.transform(df[weather_cols])
        df[soil_cols] = soil_scaler.transform(df[soil_cols])
        df[geo_cols] = geo_scaler.transform(df[geo_cols])
        df[time_cols] = numerical_scaler.transform(df[time_cols])
        
        # NOW calculate extreme_heat_flag using SCALED tmax
        # Get the threshold in scaled space
        unscaled_threshold = PREPROCESSING_METADATA.get('extreme_heat_threshold', 32.0)
        # tmax_growing_avg is at index 2 in weather_cols
        tmax_mean = weather_scaler.mean_[2]
        tmax_scale = weather_scaler.scale_[2]
        scaled_threshold = (unscaled_threshold - tmax_mean) / tmax_scale
        
        df['extreme_heat_flag'] = (df['tmax_growing_avg'] > scaled_threshold).astype(int)
        
        # Reorder columns
        final_df = df.drop(columns=['Crop', 'State', 'County'], errors='ignore')
        final_df = final_df[FEATURE_ORDER]
        
        # Predict
        prediction = model.predict(final_df)
        pred_value = float(prediction[0])
        
        # Get crop-specific info
        crop_info = CROP_UNITS.get(input_data['Crop'], 
                                    {'unit': 'Bu/Acre', 'typical_range': (10, 300)})
        min_expected, max_expected = crop_info['typical_range']

        # Determine Confidence
        r2_score = MODEL_PERFORMANCE.get(requested_model, 0.85) # Default conservatively high if unknown
        confidence_label = "Medium"
        if r2_score >= 0.90:
            confidence_label = "High"
        elif r2_score < 0.70:
            confidence_label = "Low"

        # Check against realistic ranges (Only error, no warnings for yield magnitude)
        warnings = []
        if pred_value < 0:
            warnings.append("Negative yield - check inputs")
        
        print(f"✅ Prediction: {pred_value:.2f} {crop_info['unit']} (Confidence: {confidence_label})")
        if warnings:
            for w in warnings:
                print(f"⚠️  {w}")
        print("="*80 + "\n")
        
        return jsonify({
            'prediction': round(pred_value, 2),
            'units': crop_info['unit'],
            'model': requested_model,
            'typical_range': f"{min_expected}-{max_expected} {crop_info['unit']}",
            'confidence': confidence_label,
            'confidence_score': round(r2_score, 4),
            'defaults_source': defaults_source,
            'warnings': warnings if warnings else None
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'current_model': CURRENT_MODEL,
        'available_models': list(AVAILABLE_MODELS.keys()),
        'features': len(FEATURE_ORDER)
    })
@app.route('/predict-regional', methods=['POST'])
def predict_regional():
    """
    Predict yield for all crops in a given state/county using the same input features.
    Input JSON: all fields except 'Crop', including 'State' and 'County', optional 'model'.
    Output JSON: {crop: {prediction, units, warnings}} for each crop.
    """
    data = request.json
    
    requested_model = data.get('model', CURRENT_MODEL)
    if requested_model not in AVAILABLE_MODELS:
        requested_model = CURRENT_MODEL
    model = AVAILABLE_MODELS[requested_model]['model']

    state_val = data.get('State')
    county_val = data.get('County', 'UNKNOWN').upper()
    
    if not state_val or not county_val:
        return jsonify({'error': 'State and County are required'}), 400

    results = {}
    
    # Determine Confidence
    r2_score = MODEL_PERFORMANCE.get(requested_model, 0.85)
    confidence_label = "Medium"
    if r2_score >= 0.90:
        confidence_label = "High"
    elif r2_score < 0.70:
        confidence_label = "Low"
    
    for crop_val in crop_encoder.classes_:
        # Skip crops with no yield records in this state; predictions there are extrapolation
        if VALID_COMBINATIONS and state_val not in VALID_COMBINATIONS.get(crop_val, [state_val]):
            results[crop_val] = {"not_grown": True}
            continue
        try:
            input_data = {
                'Crop': crop_val,
                'State': state_val,
                'County': county_val
            }
            
            # Apply defaults
            defaults, _ = defaults_for(state_val, county_val)
            for key, default_val in defaults.items():
                val = data.get(key)
                input_data[key] = float(val) if val not in [None, ""] else float(default_val)
            
            df = pd.DataFrame([input_data])
            
            # --- Feature Engineering ---
            df = apply_feature_engineering(df)
            
            # Encoding
            df['Crop_Encoded'] = crop_encoder.transform([crop_val])[0]
            df['State_Encoded'] = state_encoder.transform([state_val])[0]

            county_full = f"{county_val}_{state_val.upper()}"
            if county_full in county_encoder.classes_:
                df['County_Encoded'] = county_encoder.transform([county_full])[0]
            else:
                state_counties = [c for c in county_encoder.classes_ if c.endswith(f"_{state_val.upper()}")]
                df['County_Encoded'] = county_encoder.transform([state_counties[0]])[0] if state_counties else 0

            # Scale features
            weather_cols = [
                'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg',
                'prcp_growing_total', 'rh_growing_avg', 'gdd_total',
                'heat_stress_days', 'temp_range_avg',
                'temp_prcp_interaction', 'gdd_per_day', 'moisture_stress',
                'temp_variability'
            ]
            soil_cols = [
                'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct',
                'ph', 'bulk_density', 'cec', 'awc',
                'soil_quality_score', 'loam_index'
            ]
            geo_cols = ['Latitude', 'Longitude', 'lat_squared', 'long_squared', 'lat_long_interaction']
            time_cols = ['Year', 'Year_Since_Start', 'Year_Squared']

            df[weather_cols] = weather_scaler.transform(df[weather_cols])
            df[soil_cols] = soil_scaler.transform(df[soil_cols])
            df[geo_cols] = geo_scaler.transform(df[geo_cols])
            df[time_cols] = numerical_scaler.transform(df[time_cols])
            
            # Extreme heat flag
            tmax_mean = weather_scaler.mean_[2]
            tmax_scale = weather_scaler.scale_[2]
            threshold = PREPROCESSING_METADATA.get('extreme_heat_threshold', 32.0)
            scaled_threshold = (threshold - tmax_mean) / tmax_scale
            df['extreme_heat_flag'] = (df['tmax_growing_avg'] > scaled_threshold).astype(int)

            # Final feature order
            final_df = df.drop(columns=['Crop', 'State', 'County'], errors='ignore')
            final_df = final_df[FEATURE_ORDER]

            # Predict
            pred_value = float(model.predict(final_df)[0])

            crop_info = CROP_UNITS.get(crop_val, {'unit': 'Bu/Acre', 'typical_range': (10, 300)})
            
            warnings = []
            if pred_value < 0:
                warnings.append("Negative yield - check inputs")

            results[crop_val] = {
                "prediction": round(pred_value, 2),
                "units": crop_info['unit'],
                "confidence": confidence_label,
                "warnings": warnings if warnings else None
            }

        except Exception as e:
            results[crop_val] = {"error": str(e)}

    return jsonify({"model": requested_model, "results": results})

import os

@app.route("/metrics")
def metrics():
    csv_path = ARTIFACTS_DIR / 'model_comparison.csv'
    if not csv_path.exists():
        return jsonify({"error": f"CSV not found at {csv_path}"}), 404
    
    df = pd.read_csv(csv_path)
    
    numeric_cols = ['R²','RMSE','MAE','MAPE (%)','Within 10%','Training Time (s)','[BASELINE] Last Value' ,'[BASELINE] Mean' ,'[BASELINE] Moving Average' ,'[BASELINE] Median' ,]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'R²' in df.columns:
        df = df.sort_values(by='R²', ascending=False)
    
    return jsonify(df.to_dict(orient="records"))

@app.route("/feature-importance/<model_name>", methods=['GET'])
def get_feature_importance(model_name):
    """Get feature importance for a specific model"""
    try:
        # Normalize model name to match file naming
        model_file_name = model_name.lower().replace(' ', '_').replace('-', '_')
        importance_path = ARTIFACTS_DIR / 'feature_importance' / f'{model_file_name}_importance.csv'
        
        if not importance_path.exists():
            return jsonify({"error": f"Feature importance not found for {model_name}"}), 404
        
        df = pd.read_csv(importance_path)
        df = df.sort_values('Importance', ascending=False)
        
        return jsonify({
            'model': model_name,
            'features': df.to_dict(orient="records"),
            'top_10': df.head(10).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feature-importance", methods=['GET'])
def list_feature_importance():
    """List all available feature importance files"""
    try:
        importance_dir = ARTIFACTS_DIR / 'feature_importance'
        if not importance_dir.exists():
            return jsonify({"available": [], "message": "Feature importance directory not found"})
        
        files = list(importance_dir.glob('*_importance.csv'))
        available = [f.stem.replace('_importance', '').replace('_', ' ').title() for f in files]
        
        return jsonify({
            "available": available,
            "models": available
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feature-engineering", methods=['GET'])
def get_feature_engineering_info():
    """Get documentation about feature engineering"""
    return jsonify({
        "description": "13 engineered features created from raw inputs",
        "total_engineered_features": 13,
        "categories": FEATURE_ENGINEERING_INFO,
        "summary": {
            "time_features": 2,
            "climate_interactions": 4,
            "soil_features": 2,
            "geographic_features": 3
        }
    })

@app.route("/error-analysis/<model_name>", methods=['GET'])
def get_error_analysis(model_name):
    """Get detailed error analysis for a specific model"""
    try:
        # Normalize model name to match file naming
        model_file_name = model_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check in models directory (where training saves them)
        project_root = BASE_DIR.parent
        error_analysis_path = project_root / 'models' / 'error_analysis' / f'{model_file_name}_error_analysis.json'
        
        # Also check in api/artifacts if synced
        alt_path = ARTIFACTS_DIR / 'error_analysis' / f'{model_file_name}_error_analysis.json'
        
        if error_analysis_path.exists():
            with open(error_analysis_path, 'r') as f:
                return jsonify(json.load(f))
        elif alt_path.exists():
            with open(alt_path, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": f"Error analysis not found for {model_name}. Run training to generate error analysis."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/error-analysis", methods=['GET'])
def list_error_analysis():
    """List all available error analysis files"""
    try:
        project_root = BASE_DIR.parent
        error_analysis_dir = project_root / 'models' / 'error_analysis'
        
        # Also check api/artifacts
        alt_dir = ARTIFACTS_DIR / 'error_analysis'
        
        available = []
        if error_analysis_dir.exists():
            files = list(error_analysis_dir.glob('*_error_analysis.json'))
            available.extend([f.stem.replace('_error_analysis', '').replace('_', ' ').title() for f in files])
        
        if alt_dir.exists():
            files = list(alt_dir.glob('*_error_analysis.json'))
            alt_available = [f.stem.replace('_error_analysis', '').replace('_', ' ').title() for f in files]
            available.extend([a for a in alt_available if a not in available])
        
        if not available:
            return jsonify({"available": [], "message": "Error analysis directory not found. Run training to generate error analysis."})
        
        return jsonify({
            "available": available,
            "models": available
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🚀 Starting Flask Server")
    print(f"Model: {CURRENT_MODEL}")
    print(f"Port: {port}\n")
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=port)