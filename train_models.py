import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, 
    recall_score, confusion_matrix, roc_curve, log_loss,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.decomposition import PCA
import joblib

DATA_PATH = r"C:\Users\Lenovo\.gemini\antigravity\scratch\employee_attrition_project\data\employee_attrition_raw.csv"
OUTPUT_DIR = r"C:\Users\Lenovo\.gemini\antigravity\scratch\employee_attrition_project\models"
DATA_CLEAN_PATH = r"C:\Users\Lenovo\.gemini\antigravity\scratch\employee_attrition_project\data\employee_attrition.csv"
STATIC_DATA_PATH = r"C:\Users\Lenovo\.gemini\antigravity\scratch\employee_attrition_project\static\js\model_data.js"

print("--- Starting Employee Attrition Model Training Pipeline ---")

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop constant or identifier columns
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Clean and format
df.to_csv(DATA_CLEAN_PATH, index=False)
print(f"Cleaned dataset saved to {DATA_CLEAN_PATH}")

# Target variable
df['Attrition_Numeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
y = df['Attrition_Numeric'].values

# Feature Selection
selected_features = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'DistanceFromHome', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'WorkLifeBalance', 'JobInvolvement', 'StockOptionLevel',
    'OverTime', 'MaritalStatus', 'Department', 'JobRole', 'BusinessTravel', 'Gender'
]

X_raw = df[selected_features].copy()

# Encode Categorical variables
cat_encoders = {}
cat_cols = ['OverTime', 'MaritalStatus', 'Department', 'JobRole', 'BusinessTravel', 'Gender']

for col in cat_cols:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    cat_encoders[col] = {
        'classes': [str(c) for c in le.classes_],
        'mapping': {str(c): int(idx) for idx, c in enumerate(le.classes_)}
    }

feature_names = list(X_raw.columns)
X = X_raw.values

# Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

# 2. Train Supervised Models (Minimum 3, we train 4 advanced algorithms)
models = {
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=140, learning_rate=0.07, max_depth=4, 
        subsample=0.85, random_state=42
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=160, max_depth=9, criterion='gini',
        class_weight='balanced', random_state=42
    ),
    'mlp_classifier': MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        alpha=0.01, max_iter=450, random_state=42
    ),
    'svm_classifier': SVC(
        C=1.8, kernel='rbf', gamma='scale', probability=True,
        class_weight='balanced', random_state=42
    )
}

trained_models = {}
metrics_results = {}
roc_curves = {}
confusion_matrices = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Save model artifact
    model_file = os.path.join(OUTPUT_DIR, f"{name}.joblib")
    joblib.dump(model, model_file)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    loss = log_loss(y_test, y_prob)
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_all_scaled, y, cv=cv, scoring='roc_auc')
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Subsample ROC points for clean JSON export (max 25 points)
    step = max(1, len(fpr) // 25)
    roc_points = [{'fpr': round(float(f), 4), 'tpr': round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]
    if roc_points[-1] != {'fpr': 1.0, 'tpr': 1.0}:
        roc_points.append({'fpr': 1.0, 'tpr': 1.0})
    
    metrics_results[name] = {
        'display_name': name.replace('_', ' ').title(),
        'accuracy': round(float(acc) * 100, 2),
        'roc_auc': round(float(roc_auc), 4),
        'f1_score': round(float(f1), 4),
        'precision': round(float(prec) * 100, 2),
        'recall': round(float(rec) * 100, 2),
        'log_loss': round(float(loss), 4),
        'cv_auc_mean': round(float(cv_scores.mean()), 4),
        'cv_auc_std': round(float(cv_scores.std()), 4),
        'confusion_matrix': cm
    }
    roc_curves[name] = roc_points
    confusion_matrices[name] = cm
    print(f"  -> Accuracy: {acc*100:.2f}%, ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Recall: {rec*100:.2f}%")

# Save Scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

# 3. Global Feature Importance (from Gradient Boosting & Random Forest)
gb_importances = models['gradient_boosting'].feature_importances_
rf_importances = models['random_forest'].feature_importances_
avg_importances = (gb_importances * 0.6 + rf_importances * 0.4)

feature_importance_list = [
    {
        'feature': feat,
        'importance': round(float(imp) * 100, 2),
        'description': {
            'OverTime': 'Overtime work schedule (Burnout driver)',
            'MonthlyIncome': 'Monthly compensation package',
            'Age': 'Employee age / Career life-cycle stage',
            'TotalWorkingYears': 'Cumulative industry work experience',
            'YearsAtCompany': 'Company tenure and organizational commitment',
            'JobSatisfaction': 'Role & daily fulfillment rating (1-4)',
            'StockOptionLevel': 'Equity participation & retention vesting',
            'YearsWithCurrManager': 'Managerial alignment & leadership relationship',
            'EnvironmentSatisfaction': 'Workplace cultural comfort & atmosphere',
            'DistanceFromHome': 'Commute friction & daily travel strain',
            'WorkLifeBalance': 'Work-personal boundary satisfaction',
            'JobRole': 'Specific functional specialty and demand',
            'Department': 'Business unit operational pressure',
            'MaritalStatus': 'Household stability & mobility profile',
            'YearsInCurrentRole': 'Role stagnation vs. growth velocity',
            'YearsSinceLastPromotion': 'Career advancement cadence',
            'JobInvolvement': 'Personal engagement & psychological investment',
            'BusinessTravel': 'Travel frequency & dislocation',
            'Gender': 'Demographic baseline'
        }.get(feat, feat)
    }
    for feat, imp in zip(feature_names, avg_importances)
]
feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)

# 4. Unsupervised Risk Segmentation (K-Means Clustering, K=3)
print("\nTraining K-Means Unsupervised Clustering (K=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
cluster_labels = kmeans.fit_predict(X_all_scaled)
joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_model.joblib"))

sil_score = silhouette_score(X_all_scaled, cluster_labels)
db_score = davies_bouldin_score(X_all_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_all_scaled, cluster_labels)
print(f"K-Means Silhouette Score: {sil_score:.4f}, Davies-Bouldin: {db_score:.4f}")

# Cluster Profiles Analysis
df['Cluster'] = cluster_labels
cluster_profiles = []
cluster_names = [
    {"tier": "Tier 1 - Standard Low Risk", "label": "Low Risk (Core Tenured Cohort)", "color": "#10B981"},
    {"tier": "Tier 2 - Moderate / Mid-Severity", "label": "Moderate Risk (Elevated Strain)", "color": "#F59E0B"},
    {"tier": "Tier 3 - Critical Flight Risk", "label": "Severe High Risk (Burnout Flight Cohort)", "color": "#EF4444"}
]

# Order clusters by their actual attrition rate
cluster_stats = []
for c_id in range(3):
    sub = df[df['Cluster'] == c_id]
    attr_rate = sub['Attrition_Numeric'].mean()
    cluster_stats.append((c_id, attr_rate, sub))

cluster_stats.sort(key=lambda x: x[1])  # lowest to highest attrition

sorted_cluster_map = {}
for rank, (old_id, attr_rate, sub) in enumerate(cluster_stats):
    sorted_cluster_map[old_id] = rank
    meta = cluster_names[rank]
    profile = {
        'cluster_id': rank,
        'original_id': int(old_id),
        'tier': meta['tier'],
        'name': meta['label'],
        'color': meta['color'],
        'count': int(len(sub)),
        'pct_of_workforce': round(len(sub) / len(df) * 100, 1),
        'attrition_rate': round(float(attr_rate) * 100, 2),
        'mean_salary': int(round(sub['MonthlyIncome'].mean())),
        'median_salary': int(round(sub['MonthlyIncome'].median())),
        'mean_age': round(float(sub['Age'].mean()), 1),
        'mean_tenure': round(float(sub['YearsAtCompany'].mean()), 1),
        'mean_satisfaction': round(float(sub['JobSatisfaction'].mean()), 2),
        'overtime_pct': round(float((sub['OverTime'] == 'Yes').mean()) * 100, 1),
        'key_drivers': [
            "Low overtime pressure" if (sub['OverTime'] == 'Yes').mean() < 0.2 else "High overtime fatigue",
            f"Mean salary: ₹{int(round(sub['MonthlyIncome'].mean())) * 83:,} / ${int(round(sub['MonthlyIncome'].mean())):,}",
            f"Tenure: {round(float(sub['YearsAtCompany'].mean()), 1)} yrs at company",
            f"Satisfaction: {round(float(sub['JobSatisfaction'].mean()), 1)} / 4.0 rating"
        ],
        'retention_strategy': (
            "Standard performance development, bi-annual milestone reviews, competitive equity refreshes."
            if rank == 0 else
            "Mid-career retention bonuses, work-life boundary adjustments, transparent managerial 1-on-1s."
            if rank == 1 else
            "URGENT INTERVENTION: Workload rebalancing, emergency compensation adjustment, managerial realignment."
        )
    }
    cluster_profiles.append(profile)

# Remap cluster labels in DataFrame for visualization
df['Sorted_Cluster'] = df['Cluster'].map(sorted_cluster_map)

# 2D PCA for Scatter Matrix Visualization
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_all_scaled)

scatter_sample_points = []
# Sample 350 representative points for lightning-fast client render
sample_indices = np.random.RandomState(42).choice(len(df), size=350, replace=False)
for idx in sample_indices:
    row = df.iloc[idx]
    scatter_sample_points.append({
        'id': int(idx),
        'age': int(row['Age']),
        'income': int(row['MonthlyIncome']),
        'tenure': int(row['YearsAtCompany']),
        'satisfaction': int(row['JobSatisfaction']),
        'overtime': row['OverTime'],
        'attrition': row['Attrition'],
        'cluster': int(row['Sorted_Cluster']),
        'pca_x': round(float(pca_coords[idx, 0]), 3),
        'pca_y': round(float(pca_coords[idx, 1]), 3)
    })

# 5. Extract Sample Dataset Records for Data Explorer (Paginated records)
records_preview = []
for i in range(min(500, len(df))):
    r = df.iloc[i]
    records_preview.append({
        'id': int(i + 1),
        'age': int(r['Age']),
        'attrition': r['Attrition'],
        'department': r['Department'],
        'job_role': r['JobRole'],
        'monthly_income': int(r['MonthlyIncome']),
        'years_at_company': int(r['YearsAtCompany']),
        'job_satisfaction': int(r['JobSatisfaction']),
        'overtime': r['OverTime'],
        'work_life_balance': int(r['WorkLifeBalance']),
        'distance': int(r['DistanceFromHome']),
        'cluster_tier': cluster_profiles[r['Sorted_Cluster']]['tier']
    })

# 6. Pre-calculate Tipping Point Matrix (Overtime × Income Bracket)
low_inc_no_ot = df[(df['MonthlyIncome'] < 4000) & (df['OverTime'] == 'No')]['Attrition_Numeric'].mean()
low_inc_yes_ot = df[(df['MonthlyIncome'] < 4000) & (df['OverTime'] == 'Yes')]['Attrition_Numeric'].mean()
high_inc_no_ot = df[(df['MonthlyIncome'] >= 4000) & (df['OverTime'] == 'No')]['Attrition_Numeric'].mean()
high_inc_yes_ot = df[(df['MonthlyIncome'] >= 4000) & (df['OverTime'] == 'Yes')]['Attrition_Numeric'].mean()

tipping_point_matrix = {
    'non_overtime': {
        'low_income': round(float(low_inc_no_ot) * 100, 1),
        'high_income': round(float(high_inc_no_ot) * 100, 1)
    },
    'overtime': {
        'low_income': round(float(low_inc_yes_ot) * 100, 1),
        'high_income': round(float(high_inc_yes_ot) * 100, 1)
    },
    'multiplier': round(float(low_inc_yes_ot / max(0.001, low_inc_no_ot)), 2)
}

# 7. Export Model Weights & Metadata for In-Browser Engine
scaler_means = scaler.mean_.tolist()
scaler_scales = scaler.scale_.tolist()

# Extract linear approximation / tree weights for fast instant local JS inference
# For Gradient Boosting: feature importances + baseline priors
gb_weights = {
    'intercept': float(models['gradient_boosting'].init_.prior if hasattr(models['gradient_boosting'].init_, 'prior') else -1.65),
    'feature_importances': [round(float(v), 5) for v in gb_importances],
    'n_estimators': 140
}

# Master Metadata Payload
master_payload = {
    'project_title': 'Employee Attrition Prediction & Workforce Risk Intelligence System',
    'project_meta': {
        'course': 'Predictive Workforce Intelligence',
        'type': 'Enterprise Analytics',
        'dataset_source': 'IBM HR Analytics Employee Attrition & Performance Benchmark (1,470 Records)',
        'algorithms_used': [
            'Gradient Boosting Classifier (Ensemble Boosting)',
            'Random Forest Classifier (Ensemble Bagging)',
            'Multi-Layer Perceptron (Deep Neural Network)',
            'Support Vector Machine (RBF Kernel SVM)',
            'K-Means Clustering (Unsupervised Risk Segmentation)'
        ],
        'total_samples': len(df),
        'attrition_rate': round(float(df['Attrition_Numeric'].mean()) * 100, 2),
        'features_count': len(selected_features),
        'status': '100% Offline & Self-Contained'
    },
    'feature_names': feature_names,
    'cat_encoders': cat_encoders,
    'scaler': {
        'mean': [round(float(m), 5) for m in scaler_means],
        'scale': [round(float(s), 5) for s in scaler_scales]
    },
    'metrics': metrics_results,
    'roc_curves': roc_curves,
    'confusion_matrices': confusion_matrices,
    'feature_importances': feature_importance_list,
    'clustering': {
        'silhouette_score': round(float(sil_score), 4),
        'davies_bouldin_index': round(float(db_score), 4),
        'calinski_harabasz': round(float(ch_score), 2),
        'profiles': cluster_profiles,
        'scatter_points': scatter_sample_points
    },
    'tipping_point_matrix': tipping_point_matrix,
    'records_preview': records_preview,
    'dataset_summary': {
        'total_records': len(df),
        'attrited_count': int(df['Attrition_Numeric'].sum()),
        'retained_count': int(len(df) - df['Attrition_Numeric'].sum()),
        'mean_age': round(float(df['Age'].mean()), 1),
        'mean_salary_inr': int(round(df['MonthlyIncome'].mean() * 83)),
        'mean_salary_usd': int(round(df['MonthlyIncome'].mean())),
        'median_salary_inr': int(round(df['MonthlyIncome'].median() * 83)),
        'mean_tenure': round(float(df['YearsAtCompany'].mean()), 1),
        'overtime_percentage': round(float((df['OverTime'] == 'Yes').mean()) * 100, 1),
        'age_brackets': [
            {'range': '18-29 Years (Junior)', 'count': int((df['Age'] < 30).sum()), 'attrition_rate': round(float(df[df['Age'] < 30]['Attrition_Numeric'].mean()) * 100, 1)},
            {'range': '30-39 Years (Mid-Career)', 'count': int(((df['Age'] >= 30) & (df['Age'] < 40)).sum()), 'attrition_rate': round(float(df[(df['Age'] >= 30) & (df['Age'] < 40)]['Attrition_Numeric'].mean()) * 100, 1)},
            {'range': '40-49 Years (Senior)', 'count': int(((df['Age'] >= 40) & (df['Age'] < 50)).sum()), 'attrition_rate': round(float(df[(df['Age'] >= 40) & (df['Age'] < 50)]['Attrition_Numeric'].mean()) * 100, 1)},
            {'range': '50-60 Years (Executive)', 'count': int((df['Age'] >= 50).sum()), 'attrition_rate': round(float(df[df['Age'] >= 50]['Attrition_Numeric'].mean()) * 100, 1)}
        ],
        'department_distribution': [
            {'dept': 'Research & Development', 'count': int((df['Department'] == 'Research & Development').sum()), 'attrition_rate': round(float(df[df['Department'] == 'Research & Development']['Attrition_Numeric'].mean()) * 100, 1)},
            {'dept': 'Sales', 'count': int((df['Department'] == 'Sales').sum()), 'attrition_rate': round(float(df[df['Department'] == 'Sales']['Attrition_Numeric'].mean()) * 100, 1)},
            {'dept': 'Human Resources', 'count': int((df['Department'] == 'Human Resources').sum()), 'attrition_rate': round(float(df[df['Department'] == 'Human Resources']['Attrition_Numeric'].mean()) * 100, 1)}
        ]
    }
}

# Write to JSON file
with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
    json.dump(master_payload, f, indent=2)

# Write to JS file for 100% offline standalone execution without any server
js_content = f"// Automatically generated ML model metadata & precomputed weights\nconst MODEL_DATA = {json.dumps(master_payload, indent=2)};\n"
with open(STATIC_DATA_PATH, "w") as f:
    f.write(js_content)

print(f"\nAll models and metadata successfully exported!")
print(f"Metrics JSON: {os.path.join(OUTPUT_DIR, 'model_metrics.json')}")
print(f"Offline JS Data: {STATIC_DATA_PATH}")
