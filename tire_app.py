import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. PAGE SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="F1 Tire Cliff Predictor", page_icon="🛞", layout="wide")
st.title("🛞 True Tire Cliff Predictor (Fuel-Corrected)")
st.markdown("Raw lap times are deceptive because fuel weight loss makes the car faster, masking tire degradation. This tool mathematically removes the fuel advantage to reveal the true rubber degradation curve.")

# ==========================================
# 2. DATA EXTRACTION (Cached for speed)
# ==========================================
# st.cache_data ensures we don't redownload the race every time we move a slider!
@st.cache_data
def load_race_data(year, track, driver, compound):
    fastf1.Cache.enable_cache('f1_cache') # Ensure we have a folder named f1_cache
    session = fastf1.get_session(year, track, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    
    laps = session.laps.pick_driver(driver)
    # Filter for the specific tire compound and valid racing laps (Status 1)
    stint = laps[(laps['Compound'] == compound) & (laps['TrackStatus'] == '1')]
    
    df = pd.DataFrame({
        'Tire_Age': stint['TyreLife'],
        'Raw_Lap_Time': stint['LapTime'].dt.total_seconds()
    }).dropna()
    
    # Filter massive outliers (Traffic/Mistakes)
    median_time = df['Raw_Lap_Time'].median()
    df = df[df['Raw_Lap_Time'] < (median_time + 3.0)]
    
    return df

# ==========================================
# 3. SIDEBAR CONTROLS (The Strategist UI)
# ==========================================
st.sidebar.header("🔧 Pit Wall Controls")

# Data Selection
driver_input = st.sidebar.selectbox("Driver", ["HAM", "VER", "LEC", "NOR", "SAI", "ALO"])
compound_input = st.sidebar.selectbox("Tire Compound", ["MEDIUM", "HARD", "SOFT"], index=1)

st.sidebar.divider()

# Engineering Parameters
st.sidebar.subheader("Mathematical Parameters")
fuel_effect = st.sidebar.slider(
    "Fuel Weight Penalty (Sec/Lap)", 
    min_value=0.00, 
    max_value=0.15, 
    value=0.08, # Defaulting to our 0.08s calculation!
    step=0.01,
    help="How much faster does the car get per lap just from burning fuel?"
)

poly_degree = st.sidebar.slider(
    "Polynomial Degree", 
    min_value=1, 
    max_value=3, 
    value=2,
    help="1 = Linear. 2 = Single Curve (Recommended). 3 = Complex Curve."
)

# ==========================================
# 4. EXECUTE PIPELINE & DATA ENGINEERING
# ==========================================
try:
    with st.spinner("Downloading F1 Telemetry..."):
        df = load_race_data(2023, 'Spain', driver_input, compound_input)
except Exception as e:
    st.error("Could not load data for this specific combination. Try a different compound.")
    st.stop()

if df.empty or len(df) < 5:
    st.warning("Not enough clean laps in this stint to build a reliable Machine Learning model.")
    st.stop()

# THE SECRET SAUCE: Fuel Correction Math
# We ADD the time back based on how many laps the tire has done.
df['Corrected_Lap_Time'] = df['Raw_Lap_Time'] + (df['Tire_Age'] * fuel_effect)

# ==========================================
# 5. MACHINE LEARNING MODELS
# ==========================================
X = df[['Tire_Age']]
y_raw = df['Raw_Lap_Time']
y_corrected = df['Corrected_Lap_Time']

# Build a smooth X-axis for drawing the curves
X_smooth = pd.DataFrame({'Tire_Age': np.linspace(X['Tire_Age'].min(), X['Tire_Age'].max() + 5, 100)})

# Model 1: Train on Raw Data (The Deceptive Model)
model_raw = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
model_raw.fit(X, y_raw)
y_raw_pred = model_raw.predict(X_smooth)

# Model 2: Train on Corrected Data (The True Model)
model_corrected = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
model_corrected.fit(X, y_corrected)
y_corrected_pred = model_corrected.predict(X_smooth)

# ==========================================
# 6. VISUALIZATION
# ==========================================
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="darkgrid")

    # Plot 1: Raw Data (Deceptive)
    ax.scatter(df['Tire_Age'], df['Raw_Lap_Time'], color='gray', alpha=0.5, label="Raw Lap Times (With Fuel)")
    ax.plot(X_smooth['Tire_Age'], y_raw_pred, color='gray', linestyle='--', label="Deceptive ML Curve")

    # Plot 2: Corrected Data (The Truth)
    ax.scatter(df['Tire_Age'], df['Corrected_Lap_Time'], color='#00d2be', s=80, edgecolor='black', label="True Lap Times (Fuel Removed)")
    ax.plot(X_smooth['Tire_Age'], y_corrected_pred, color='red', linewidth=3, label="True Tire Degradation Curve")

    ax.set_title(f'Tire Performance Profile: {driver_input} ({compound_input})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tire Age (Laps)', fontsize=12)
    ax.set_ylabel('Lap Time (Seconds)', fontsize=12)
    ax.legend()
    
    st.pyplot(fig)

with col2:
    st.subheader("Model Insights")
    st.info(f"**Baseline Pace:** {df['Corrected_Lap_Time'].min():.2f}s")
    
    # Calculate the exact lap the true curve starts going up
    min_idx = np.argmin(y_corrected_pred)
    peak_lap = X_smooth['Tire_Age'].iloc[min_idx]
    
    st.success(f"**Peak Performance:** Lap {peak_lap:.1f}")
    
    # Find when the time drops by 1.5s from the peak
    cliff_threshold = y_corrected_pred.min() + 1.5
    cliff_indices = np.where(y_corrected_pred[min_idx:] >= cliff_threshold)[0]
    
    if len(cliff_indices) > 0:
        cliff_age = X_smooth['Tire_Age'].iloc[min_idx + cliff_indices[0]]
        st.error(f"**Predicted Cliff:** Lap {cliff_age:.1f}")
    else:
        st.warning("**Predicted Cliff:** Not reached in this stint.")