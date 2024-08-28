#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import resample
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[3]:


# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
file_path = 'C:\\Users\\aasmi\\OneDrive\\Documents\\GitHub\\openbiomechanics\\baseball_pitching\\data\\poi\\poi_metrics.csv'
poi_metrics = pd.read_csv(file_path)
poi_metrics = poi_metrics.dropna()

# Define kinematic and kinetic variables
kinematic_variables = [
    "max_shoulder_internal_rotational_velo", "max_elbow_extension_velo", "max_torso_rotational_velo",
    "max_rotation_hip_shoulder_separation","max_elbow_flexion","max_shoulder_external_rotation",
    "elbow_flexion_fp","elbow_pronation_fp","rotation_hip_shoulder_separation_fp","shoulder_horizontal_abduction_fp",
    "shoulder_abduction_fp","shoulder_external_rotation_fp","lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br","lead_knee_extension_angular_velo_max","torso_anterior_tilt_fp",
    "torso_lateral_tilt_fp","torso_rotation_fp","pelvis_anterior_tilt_fp","pelvis_lateral_tilt_fp","pelvis_rotation_fp",
    "max_cog_velo_x","torso_rotation_min","max_pelvis_rotational_velo","glove_shoulder_horizontal_abduction_fp",
    "glove_shoulder_abduction_fp","glove_shoulder_external_rotation_fp","glove_shoulder_abduction_mer","elbow_flexion_mer",
    "torso_anterior_tilt_mer","torso_lateral_tilt_mer","torso_rotation_mer","torso_anterior_tilt_br","torso_lateral_tilt_br",
    "torso_rotation_br", "lead_knee_extension_from_fp_to_br", "cog_velo_pkh", "stride_length", "stride_angle",
    "arm_slot", "timing_peak_torso_to_peak_pelvis_rot_velo", "max_shoulder_horizontal_abduction"
]

kinetic_variables = [
    "shoulder_transfer_fp_br", "shoulder_generation_fp_br",
    "shoulder_absorption_fp_br", "elbow_transfer_fp_br", "elbow_generation_fp_br",
    "elbow_absorption_fp_br", "lead_hip_transfer_fp_br", "lead_hip_generation_fp_br",
    "lead_hip_absorption_fp_br", "lead_knee_transfer_fp_br", "lead_knee_generation_fp_br",
    "lead_knee_absorption_fp_br", "rear_hip_transfer_pkh_fp", "rear_hip_generation_pkh_fp",
    "rear_hip_absorption_pkh_fp", "rear_knee_transfer_pkh_fp", "rear_knee_generation_pkh_fp",
    "rear_knee_absorption_pkh_fp", "pelvis_lumbar_transfer_fp_br",
    "thorax_distal_transfer_fp_br", "rear_grf_x_max", "rear_grf_y_max", "rear_grf_z_max",
    "rear_grf_mag_max", "lead_grf_x_max", "lead_grf_y_max",
    "lead_grf_z_max", "lead_grf_mag_max",'elbow_varus_moment','shoulder_internal_rotation_moment'
]

# Prepare data
kinematic_df = poi_metrics[kinematic_variables + ['pitch_speed_mph']].copy()
kinetic_df = poi_metrics[kinetic_variables + ['pitch_speed_mph']].copy()

# 1. Spearman Correlation Analysis
def spearman_correlation_analysis(X, y):
    correlations = []
    for column in X.columns:
        corr, _ = spearmanr(X[column], y)
        correlations.append((column, corr))
    return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

print("Performing Spearman correlation analysis for kinematic variables...")
kinematic_correlations = spearman_correlation_analysis(kinematic_df[kinematic_variables], kinematic_df['pitch_speed_mph'])
print("Top 10 correlated kinematic variables:")
for var, corr in kinematic_correlations[:10]:
    print(f"{var}: {corr:.4f}")

print("\nPerforming Spearman correlation analysis for kinetic variables...")
kinetic_correlations = spearman_correlation_analysis(kinetic_df[kinetic_variables], kinetic_df['pitch_speed_mph'])
print("Top 10 correlated kinetic variables:")
for var, corr in kinetic_correlations[:10]:
    print(f"{var}: {corr:.4f}")

# 2. Bootstrapped Feature Importance
def bootstrapped_feature_importance(X, y, n_iterations=100):
    importances = []
    for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
        X_boot, y_boot = resample(X, y)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_boot, y_boot)
        importances.append(rf.feature_importances_)
    return np.mean(importances, axis=0), np.std(importances, axis=0)

print("\nPerforming bootstrapped feature importance for kinematic variables...")
kinematic_importances, kinematic_std = bootstrapped_feature_importance(kinematic_df[kinematic_variables], kinematic_df['pitch_speed_mph'])
kinematic_importance_ranking = sorted(zip(kinematic_variables, kinematic_importances), key=lambda x: x[1], reverse=True)
print("Top 10 important kinematic variables:")
for var, imp in kinematic_importance_ranking[:10]:
    print(f"{var}: {imp:.4f}")

print("\nPerforming bootstrapped feature importance for kinetic variables...")
kinetic_importances, kinetic_std = bootstrapped_feature_importance(kinetic_df[kinetic_variables], kinetic_df['pitch_speed_mph'])
kinetic_importance_ranking = sorted(zip(kinetic_variables, kinetic_importances), key=lambda x: x[1], reverse=True)
print("Top 10 important kinetic variables:")
for var, imp in kinetic_importance_ranking[:10]:
    print(f"{var}: {imp:.4f}")

# 3. Gradient Boosting and Random Forest Models
def train_and_evaluate_model(X, y, model_type="RandomForest"):
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    
    return feature_importance

print("\nTraining Random Forest model for kinematic variables...")
rf_kinematic_importance = train_and_evaluate_model(kinematic_df[kinematic_variables], kinematic_df['pitch_speed_mph'], "RandomForest")
print("Top 10 important kinematic variables (Random Forest):")
for var, imp in rf_kinematic_importance[:10]:
    print(f"{var}: {imp:.4f}")

print("\nTraining Gradient Boosting model for kinematic variables...")
gb_kinematic_importance = train_and_evaluate_model(kinematic_df[kinematic_variables], kinematic_df['pitch_speed_mph'], "GradientBoosting")
print("Top 10 important kinematic variables (Gradient Boosting):")
for var, imp in gb_kinematic_importance[:10]:
    print(f"{var}: {imp:.4f}")

print("\nTraining Random Forest model for kinetic variables...")
rf_kinetic_importance = train_and_evaluate_model(kinetic_df[kinetic_variables], kinetic_df['pitch_speed_mph'], "RandomForest")
print("Top 10 important kinetic variables (Random Forest):")
for var, imp in rf_kinetic_importance[:10]:
    print(f"{var}: {imp:.4f}")

print("\nTraining Gradient Boosting model for kinetic variables...")
gb_kinetic_importance = train_and_evaluate_model(kinetic_df[kinetic_variables], kinetic_df['pitch_speed_mph'], "GradientBoosting")
print("Top 10 important kinetic variables (Gradient Boosting):")
for var, imp in gb_kinetic_importance[:10]:
    print(f"{var}: {imp:.4f}")

# Summary of findings
print("\nSummary of Important Kinematic Factors:")
print("Top 5 by Correlation:")
for var, corr in kinematic_correlations[:5]:
    print(f"{var}: {corr:.4f}")
print("\nTop 5 by Bootstrapped Random Forest:")
for var, imp in kinematic_importance_ranking[:5]:
    print(f"{var}: {imp:.4f}")
print("\nTop 5 by Random Forest:")
for var, imp in rf_kinematic_importance[:5]:
    print(f"{var}: {imp:.4f}")
print("\nTop 5 by Gradient Boosting:")
for var, imp in gb_kinematic_importance[:5]:
    print(f"{var}: {imp:.4f}")

print("\nSummary of Important Kinetic Factors:")
print("Top 5 by Correlation:")
for var, corr in kinetic_correlations[:5]:
    print(f"{var}: {corr:.4f}")
print("\nTop 5 by Bootstrapped Random Forest:")
for var, imp in kinetic_importance_ranking[:5]:
    print(f"{var}: {imp:.4f}")
print("\nTop 5 by Random Forest:")
for var, imp in rf_kinetic_importance[:5]:
    print(f"{var}: {imp:.4f}")
print("\nTop 5 by Gradient Boosting:")
for var, imp in gb_kinetic_importance[:5]:
    print(f"{var}: {imp:.4f}")

print("\nAnalysis complete. Please review the outputs to identify the most consistently important factors across different methods.")


# In[5]:


# Function to categorize pitch speeds
def categorize_pitch_speed(speed):
    if speed >= 90:
        return '90+ mph'
    elif 85 <= speed < 90:
        return '85-90 mph'
    elif 80 <= speed < 85:
        return '80-85 mph'
    else:
        return 'Less than 80 mph'

# Apply categorization
poi_metrics['speed_category'] = poi_metrics['pitch_speed_mph'].apply(categorize_pitch_speed)

# Common important kinematic features
common_kinematic_features = [
    "max_shoulder_horizontal_abduction",
    "shoulder_horizontal_abduction_fp",
    "max_shoulder_internal_rotational_velo",
    "max_torso_rotational_velo",
    "max_rotation_hip_shoulder_separation"
]

# Common important kinetic features
common_kinetic_features = [
    "elbow_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
    "shoulder_transfer_fp_br",
    "lead_knee_absorption_fp_br",
    "rear_knee_absorption_pkh_fp"
]

# Function to perform ANOVA and Tukey's HSD
def perform_anova_and_tukey(df, var):
    model = ols(f'{var} ~ C(speed_category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Perform Tukey's HSD
    tukey = pairwise_tukeyhsd(df[var], df['speed_category'])
    
    return anova_table, tukey

# Function to plot boxplot for each variable
def plot_boxplot(df, var):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='speed_category', y=var, data=df, 
                order=['Less than 80 mph', '80-85 mph', '85-90 mph', '90+ mph'])
    plt.title(f'{var} across Pitch Speed Categories')
    plt.xlabel('Pitch Speed Category')
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

# Perform ANOVA and Tukey's HSD for kinematic features
print("ANOVA and Tukey's HSD for Common Important Kinematic Features:")
for feature in common_kinematic_features:
    anova_result, tukey_result = perform_anova_and_tukey(poi_metrics, feature)
    print(f"\nANOVA results for {feature}:")
    print(anova_result)
    print(f"\nTukey's HSD results for {feature}:")
    print(tukey_result)
    plot_boxplot(poi_metrics, feature)

# Perform ANOVA and Tukey's HSD for kinetic features
print("\nANOVA and Tukey's HSD for Common Important Kinetic Features:")
for feature in common_kinetic_features:
    anova_result, tukey_result = perform_anova_and_tukey(poi_metrics, feature)
    print(f"\nANOVA results for {feature}:")
    print(anova_result)
    print(f"\nTukey's HSD results for {feature}:")
    print(tukey_result)
    plot_boxplot(poi_metrics, feature)

print("\nAnalysis complete. Please review the ANOVA results, Tukey's HSD outcomes, and boxplots for each feature.")


# In[7]:


# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
file_path = 'C:\\Users\\aasmi\\OneDrive\\Documents\\GitHub\\openbiomechanics\\baseball_pitching\\data\\poi\\poi_metrics.csv'
poi_metrics = pd.read_csv(file_path)

# Drop rows with any NaN values
poi_metrics = poi_metrics.dropna()

# Display the number of columns and rows
print(f"Number of rows: {poi_metrics.shape[0]}")
print(f"Number of columns: {poi_metrics.shape[1]}")

# Optional: Display detailed information about the DataFrame
print(poi_metrics.info())


# In[ ]:




