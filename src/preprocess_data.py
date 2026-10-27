#!/usr/bin/env python3
"""
Leakage-Safe Preprocessing Pipeline for Crop Yield Prediction
=============================================================
- Temporal split first
- Fit encoders and scalers only on training data
- Transform validation and test sets safely
- Saves X_train, X_val, X_test, y_train, y_val, y_test, and preprocessing artifacts
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CropYieldPreprocessor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.artifacts_dir = self.processed_dir / 'preprocessing_artifacts'
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize encoders and scalers
        self.crop_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.county_encoder = LabelEncoder()
        self.numerical_scaler = RobustScaler()
        self.weather_scaler = StandardScaler()
        self.soil_scaler = StandardScaler()
        self.geo_scaler = StandardScaler()
        
        self.feature_columns = {}
        self.metadata = {}
        self.crop_stats = {}

    # ----------------- Helper functions -----------------
    def convert_numpy(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # ----------------- Data loading -----------------
    def load_and_validate(self):
        filepath = self.processed_dir / 'master_yield_dataset.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Master dataset not found at {filepath}")
        df = pd.read_csv(filepath)
        assert 'Yield' in df.columns, "Target 'Yield' not found"
        return df

    def handle_missing_values(self, df):
        # Drop outcome variables
        for col in ['AREA_HARVESTED', 'PRODUCTION']:
            if col in df.columns:
                df = df.drop(columns=[col])
        # Drop rows with missing target
        df = df[df['Yield'].notna()]
        # Drop any remaining missing rows
        df = df.dropna()
        return df

    # ----------------- Feature engineering -----------------
    def create_features(self, df):
        df['Year_Since_Start'] = df['Year'] - df['Year'].min()
        df['Year_Squared'] = df['Year'] ** 2
        df['temp_prcp_interaction'] = df['tavg_growing_avg'] * df['prcp_growing_total']
        df['gdd_per_day'] = df['gdd_total'] / 180
        df['moisture_stress'] = df['prcp_growing_total'] / (df['tavg_growing_avg'] + 1)
        df['soil_quality_score'] = (
            df['organic_matter_pct'] * 0.3 +
            (7 - abs(df['ph'] - 6.5)) * 0.2 +
            df['awc'] * 100 * 0.3 +
            df['cec'] * 0.2
        )
        df['loam_index'] = 100 - (abs(df['clay_pct'] - 20) + abs(df['sand_pct'] - 40) + abs(df['silt_pct'] - 40))
        df['temp_variability'] = df['temp_range_avg'] * df['heat_stress_days']
        df['extreme_heat_flag'] = (df['heat_stress_days'] > df.groupby('Crop')['heat_stress_days'].transform('median')).astype(int)
        df['lat_squared'] = df['Latitude'] ** 2
        df['long_squared'] = df['Longitude'] ** 2
        df['lat_long_interaction'] = df['Latitude'] * df['Longitude']
        return df

    def compute_crop_statistics(self, df):
        for crop in df['Crop'].unique():
            crop_data = df[df['Crop'] == crop]
            self.crop_stats[crop] = {
                'mean_yield': float(crop_data['Yield'].mean()),
                'std_yield': float(crop_data['Yield'].std()),
                'median_yield': float(crop_data['Yield'].median()),
                'min_yield': float(crop_data['Yield'].min()),
                'max_yield': float(crop_data['Yield'].max()),
                'count': int(len(crop_data))
            }

    # ----------------- Temporal split -----------------
    def temporal_split(self, df, test_years=3, val_years=2):
        unique_years = sorted(df['Year'].unique())
        train_years = unique_years[:-test_years-val_years]
        val_years_list = unique_years[-test_years-val_years:-test_years]
        test_years_list = unique_years[-test_years:]
        train_df = df[df['Year'].isin(train_years)].copy()
        val_df = df[df['Year'].isin(val_years_list)].copy()
        test_df = df[df['Year'].isin(test_years_list)].copy()
        self.metadata['split_info'] = {
            'train_years': [int(y) for y in train_years],
            'val_years': [int(y) for y in val_years_list],
            'test_years': [int(y) for y in test_years_list]
        }
        return train_df, val_df, test_df

    # ----------------- Encoding -----------------
    def encode_categorical(self, df, fit=False):
        # Crop
        if fit:
            df['Crop_Encoded'] = self.crop_encoder.fit_transform(df['Crop'])
            self.metadata['crop_classes'] = self.crop_encoder.classes_.tolist()
        else:
            df['Crop_Encoded'] = self.crop_encoder.transform(df['Crop'])
        # State
        if fit:
            df['State_Encoded'] = self.state_encoder.fit_transform(df['State'])
            self.metadata['state_classes'] = self.state_encoder.classes_.tolist()
        else:
            df['State_Encoded'] = self.state_encoder.transform(df['State'])
        # County
        df['County_State'] = df['County'] + '_' + df['State']
        if fit:
            df['County_Encoded'] = self.county_encoder.fit_transform(df['County_State'])
            self.metadata['county_classes'] = self.county_encoder.classes_.tolist()
        else:
            known_counties = set(self.county_encoder.classes_)
            df['County_Encoded'] = df['County_State'].apply(
                lambda x: self.county_encoder.transform([x])[0] if x in known_counties else -1
            )
        df.drop(columns=['County_State'], inplace=True)
        return df

    # ----------------- Scaling -----------------
    def scale_features(self, df, fit=False):
        weather_features = [
            'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg',
            'prcp_growing_total', 'rh_growing_avg', 'gdd_total',
            'heat_stress_days', 'temp_range_avg',
            'temp_prcp_interaction', 'gdd_per_day', 'moisture_stress',
            'temp_variability'
        ]
        soil_features = [
            'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct',
            'ph', 'bulk_density', 'cec', 'awc',
            'soil_quality_score', 'loam_index'
        ]
        geo_features = [
            'Latitude', 'Longitude', 'lat_squared', 'long_squared',
            'lat_long_interaction'
        ]
        time_features = ['Year', 'Year_Since_Start', 'Year_Squared']
        self.feature_columns['weather'] = weather_features
        self.feature_columns['soil'] = soil_features
        self.feature_columns['geo'] = geo_features
        self.feature_columns['time'] = time_features
        # Scale
        if fit:
            df[weather_features] = self.weather_scaler.fit_transform(df[weather_features])
            df[soil_features] = self.soil_scaler.fit_transform(df[soil_features])
            df[geo_features] = self.geo_scaler.fit_transform(df[geo_features])
            df[time_features] = self.numerical_scaler.fit_transform(df[time_features])
        else:
            df[weather_features] = self.weather_scaler.transform(df[weather_features])
            df[soil_features] = self.soil_scaler.transform(df[soil_features])
            df[geo_features] = self.geo_scaler.transform(df[geo_features])
            df[time_features] = self.numerical_scaler.transform(df[time_features])
        return df

    # ----------------- Prepare X, y -----------------
    def prepare_Xy(self, df):
        exclude_cols = ['Yield', 'Crop', 'State', 'County', 'State_Name', 'STATE_FIPS_CODE', 'COUNTY_CODE']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['Yield']
        self.feature_columns['final_features'] = X.columns.tolist()
        self.metadata['n_features'] = X.shape[1]
        return X, y

    # ----------------- Save artifacts and datasets -----------------
    def save_artifacts(self):
        artifacts = {
            'crop_encoder': self.crop_encoder,
            'state_encoder': self.state_encoder,
            'county_encoder': self.county_encoder,
            'weather_scaler': self.weather_scaler,
            'soil_scaler': self.soil_scaler,
            'geo_scaler': self.geo_scaler,
            'numerical_scaler': self.numerical_scaler
        }
        for name, obj in artifacts.items():
            with open(self.artifacts_dir / f'{name}.pkl', 'wb') as f:
                pickle.dump(obj, f)
        with open(self.artifacts_dir / 'feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        with open(self.artifacts_dir / 'crop_stats.json', 'w') as f:
            json.dump(self.crop_stats, f, indent=2)
        with open(self.artifacts_dir / 'metadata.json', 'w') as f:
            json.dump(self.convert_numpy(self.metadata), f, indent=2)

    def save_valid_combinations(self, df):
        """Record which states actually report yields for each crop.

        The API uses this to refuse predictions for crops a state does not grow,
        which the models would otherwise extrapolate wildly.
        """
        combos = {
            crop: sorted(states.unique().tolist())
            for crop, states in df.groupby('Crop')['State']
        }
        with open(self.artifacts_dir / 'valid_crop_state_combinations.json', 'w') as f:
            json.dump(combos, f, indent=2)

    # Raw inputs the API fills in when the user leaves them blank
    COUNTY_DEFAULT_COLUMNS = [
        'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg', 'prcp_growing_total',
        'rh_growing_avg', 'gdd_total', 'heat_stress_days', 'temp_range_avg',
        'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct', 'ph', 'bulk_density',
        'cec', 'awc', 'Latitude', 'Longitude',
    ]

    def save_county_defaults(self, train_df):
        """Per-county location, soil and average growing-season weather from the
        training years, so a prediction for a county uses that county's conditions."""
        cols = [c for c in self.COUNTY_DEFAULT_COLUMNS if c in train_df.columns]
        keys = train_df['County'].str.upper() + '_' + train_df['State']
        means = train_df[cols].groupby(keys).mean().round(3)
        with open(self.artifacts_dir / 'county_defaults.json', 'w') as f:
            json.dump(means.to_dict(orient='index'), f)

    def save_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        datasets = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        for name, data in datasets.items():
            with open(self.artifacts_dir / f'{name}.pkl', 'wb') as f:
                pickle.dump(data, f)

    # ----------------- Full pipeline -----------------
    def run_full_pipeline(self, test_years=3, val_years=2):
        print("Loading raw dataset...")
        df = self.load_and_validate()
        df = self.handle_missing_values(df)
        df = self.create_features(df)
        self.compute_crop_statistics(df)
        self.save_valid_combinations(df)

        # ---------- Temporal split first ----------
        train_df, val_df, test_df = self.temporal_split(df, test_years=test_years, val_years=val_years)
        self.save_county_defaults(train_df)

        # ---------- Encode categorical ----------
        train_df = self.encode_categorical(train_df, fit=True)
        val_df = self.encode_categorical(val_df, fit=False)
        test_df = self.encode_categorical(test_df, fit=False)

        # ---------- Scale ----------
        train_df = self.scale_features(train_df, fit=True)
        val_df = self.scale_features(val_df, fit=False)
        test_df = self.scale_features(test_df, fit=False)

        # ---------- Prepare X, y ----------
        X_train, y_train = self.prepare_Xy(train_df)
        X_val, y_val = self.prepare_Xy(val_df)
        X_test, y_test = self.prepare_Xy(test_df)

        # ---------- Save artifacts and datasets ----------
        self.save_artifacts()
        self.save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)

        print("\n✅ Preprocessing complete!")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------- Run -----------------
if __name__ == '__main__':
    preprocessor = CropYieldPreprocessor()
    preprocessor.run_full_pipeline()
