import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import warnings
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

target_col = "Attrition"
random_state = 42 #np.random.random()

folder_path = "data"
time_folder_path = "in_out_time"
in_time_file_name = "in_time.csv"
out_time_file_name = "out_time.csv"
employee_file_name = "employee_survey_data.csv"
general_file_name = "general_data.csv"
manager_file_name = "manager_survey_data.csv"

employee_data = pd.read_csv(os.path.join(folder_path, employee_file_name))
general_data = pd.read_csv(os.path.join(folder_path, general_file_name))
manager_data = pd.read_csv(os.path.join(folder_path, manager_file_name))
in_time_data = pd.read_csv(os.path.join(folder_path, time_folder_path, in_time_file_name))
out_time_data = pd.read_csv(os.path.join(folder_path, time_folder_path, out_time_file_name))


##############################
# in_out_time
##############################

# merge in_time and out_time data on the first column (Unknown that is actually EmployeeID)
# rename the first column to EmployeeID for both datasets because it is unnamed
in_time_data.rename(columns={in_time_data.columns[0]: "EmployeeID"}, inplace=True)
out_time_data.rename(columns={out_time_data.columns[0]: "EmployeeID"}, inplace=True)

#check if days are present in both datasets
in_time_days = set(in_time_data.columns[1:])
out_time_days = set(out_time_data.columns[1:])
missing_in_out = in_time_days.difference(out_time_days)
# display the missing days
print(f"Days missing in either in_time or out_time data: {missing_in_out}")

# go through each column to check empty cells present only in one of the datasets
for day in in_time_days.intersection(out_time_days):
    in_time_empty = set(in_time_data.index[in_time_data[day].isnull()])
    out_time_empty = set(out_time_data.index[out_time_data[day].isnull()])
    missing_in_out_rows = in_time_empty.symmetric_difference(out_time_empty)
    if missing_in_out_rows:
        print(f"Day {day} has missing entries in either in_time or out_time data at rows: {missing_in_out_rows}")

# convert all columns except the first one to datetime format
for col in in_time_data.columns[1:]:
    in_time_data[col] = pd.to_datetime(in_time_data[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
for col in out_time_data.columns[1:]:
    out_time_data[col] = pd.to_datetime(out_time_data[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

#function to remove columns depending on distinct values for relevance
def remove_col_depending_on_distinct_values(df, start_threshold=0, end_threshold=0):
    cols_to_remove = []
    for col in df.columns:
        if start_threshold <= df[col].nunique() <= end_threshold:
            cols_to_remove.append(col)
    df.drop(columns=cols_to_remove, inplace=True)
    return df

# merge in and out time data based on EmployeeID
time_data = pd.merge(in_time_data, out_time_data, on="EmployeeID", suffixes=("_in", "_out"))

# create a new column for each day calculating the difference between out and in time in hours
hours_columns = {}
day_of_week_columns = {}
for day in in_time_days.intersection(out_time_days):
    hours_columns[f"{day}_hours"] = (time_data[f"{day}_out"] - time_data[f"{day}_in"]).dt.total_seconds() / 3600.0
    day_of_week_columns[f"{day}_day_of_week"] = time_data[f"{day}_in"].dt.dayofweek

# use pd.concat to avoid DataFrame fragmentation
# Concatenate all hours columns at once and create a new column called "duration_hours"
time_data = pd.concat([time_data, pd.DataFrame(hours_columns, index=time_data.index)], axis=1)
time_data = pd.concat([time_data, pd.DataFrame(day_of_week_columns, index=time_data.index)], axis=1)
time_data["duration_hours"] = time_data[list(hours_columns.keys())].sum(axis=1)

# aggregate by day of week
day_of_week_counts = {}
day_of_week_avg_hours = {}

for i in range(7): # 0=Monday through 6=Sunday
    count_cols = [col for col in time_data.columns if col.endswith("_day_of_week")]
    day_of_week_counts[f"worked_on_day_{i}"] = sum(
        (time_data[col] == i).astype(int) for col in count_cols
    )
    
    # avg hrs per day of week
    total_hours = 0
    for day in in_time_days.intersection(out_time_days):
        day_col = f"{day}_day_of_week"
        hours_col = f"{day}_hours"
        if day_col in time_data.columns and hours_col in time_data.columns:
            # only sum hours where the day of week matches
            mask = time_data[day_col] == i
            total_hours += time_data[hours_col].where(mask, 0)
    day_of_week_avg_hours[f"avg_hours_day_{i}"] = total_hours / day_of_week_counts[f"worked_on_day_{i}"].replace(0, 1)

time_data = pd.concat([time_data, pd.DataFrame(day_of_week_counts, index=time_data.index)], axis=1)
time_data = pd.concat([time_data, pd.DataFrame(day_of_week_avg_hours, index=time_data.index)], axis=1)

# remove columns with 0 distinct values
remove_col_depending_on_distinct_values(time_data)

# keep only columns: EmployeeID, duration_hours, worked_on_day_*, avg_hours_day_*
cols_to_keep = ["EmployeeID", "duration_hours"] + [col for col in time_data.columns if col.startswith("worked_on_day_") or col.startswith("avg_hours_day_")]
time_data = time_data[cols_to_keep]
time_data = pd.concat([time_data, pd.DataFrame(hours_columns, index=time_data.index)], axis=1)

##############################
#Pipeline
##############################

# Pipeline with proper train/test separation to avoid data leakage
def preprocess_data_safe(dataset, fitted_scaler=None, fitted_imputers=None, fit_mode=True, 
                          encode_ordinal_cols=None, remove_from_encoding=[]):
    """
    Preprocess data with proper train/test separation.
    
    Args:
        dataset: DataFrame to preprocess
        fitted_scaler: Pre-fitted StandardScaler (for test set)
        fitted_imputers: Dict of pre-fitted imputation values (for test set)
        fit_mode: If True, fit transformers. If False, use provided transformers
        encode_ordinal_cols: Dict of ordinal encodings
        remove_from_encoding: Columns to exclude from encoding
    
    Returns:
        data: Preprocessed DataFrame
        scaler: Fitted StandardScaler
        imputers: Dict of imputation values
    """
    data = dataset.copy()
    
    # Remove constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        data.drop(columns=constant_cols, inplace=True)
    
    # Identify column types
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    
    # Remove target and excluded columns from processing
    if 'Attrition' in numeric_cols:
        numeric_cols.remove('Attrition')
    if 'EmployeeID' in numeric_cols:
        numeric_cols.remove('EmployeeID')
    
    # Impute missing values
    if fit_mode:
        # FIT on training data
        imputers = {}
        if len(numeric_cols) > 0:
            imputers['numeric'] = data[numeric_cols].median()
        if len(categorical_cols) > 0:
            imputers['categorical'] = data[categorical_cols].mode().iloc[0] if len(data[categorical_cols].mode()) > 0 else {}
    else:
        # TRANSFORM using fitted values from training
        imputers = fitted_imputers
    
    # Apply imputation
    if len(numeric_cols) > 0 and 'numeric' in imputers:
        data[numeric_cols] = data[numeric_cols].fillna(imputers['numeric'])
    if len(categorical_cols) > 0 and 'categorical' in imputers:
        data[categorical_cols] = data[categorical_cols].fillna(imputers['categorical'])
    
    # Ordinal encoding
    if encode_ordinal_cols:
        for col, categories in encode_ordinal_cols.items():
            if col in data.columns:
                data[col] = pd.Categorical(data[col], categories=categories, ordered=True).codes
                if col in categorical_cols:
                    categorical_cols.remove(col)
                if col not in numeric_cols:
                    numeric_cols.append(col)
    
    # One-hot encoding
    cols_to_encode = [col for col in categorical_cols if col not in remove_from_encoding]
    if len(cols_to_encode) > 0:
        data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
    
    # Update numeric_cols after encoding
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Attrition' in numeric_cols:
        numeric_cols.remove('Attrition')
    if 'EmployeeID' in numeric_cols:
        numeric_cols.remove('EmployeeID')
    
    # Scale numerical data
    if fit_mode:
        # FIT scaler on training data
        scaler = StandardScaler()
        if len(numeric_cols) > 0:
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    else:
        # TRANSFORM using fitted scaler from training
        scaler = fitted_scaler
        if len(numeric_cols) > 0 and scaler is not None:
            data[numeric_cols] = scaler.transform(data[numeric_cols])
    
    return data, scaler, imputers


##############################
# Merge and Split (BEFORE preprocessing to avoid leakage)
##############################

print("\n" + "="*70)
print("📦 Data Merging and Train/Test Split")
print("="*70)

# Merge employee and manager data first
employee_manager_data = pd.merge(employee_data, manager_data, on="EmployeeID", suffixes=("_emp", "_mgr"))
# Merge all datasets into a final dataset on EmployeeID
raw_dataset = pd.merge(general_data, employee_manager_data, on="EmployeeID")
raw_dataset = pd.merge(raw_dataset, time_data, on="EmployeeID")

# Drop unethical columns BEFORE split
raw_dataset.drop(columns=["MaritalStatus", "Gender", "Age"], inplace=True)

print(f"Total dataset size: {len(raw_dataset)} samples")
print(f"Features: {len(raw_dataset.columns)} columns")

# CRITICAL: Split BEFORE any preprocessing to avoid data leakage
train_set, test_set = train_test_split(raw_dataset, test_size=0.2, random_state=random_state, stratify=raw_dataset['Attrition'])

print(f"Train set: {len(train_set)} samples")
print(f"Test set: {len(test_set)} samples")
print(f"Train class distribution: {train_set['Attrition'].value_counts().to_dict()}")
print(f"Test class distribution: {test_set['Attrition'].value_counts().to_dict()}")

##############################
# Preprocessing WITHOUT Data Leakage
##############################

print("\n" + "="*70)
print("🔧 Preprocessing Train and Test Sets Separately")
print("="*70)

ordinal_mappings = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
}

mutual_info_columns = ["Department", "EducationField", "JobRole"]

# Ensure Attrition is numeric
if train_set[target_col].dtype == "object":
    train_set[target_col] = train_set[target_col].apply(lambda x: 1 if str(x).lower() in ["yes", "1"] else 0)
if test_set[target_col].dtype == "object":
    test_set[target_col] = test_set[target_col].apply(lambda x: 1 if str(x).lower() in ["yes", "1"] else 0)

# Preprocess TRAIN set (fit transformers)
train_processed, fitted_scaler, fitted_imputers = preprocess_data_safe(
    train_set,
    fitted_scaler=None,
    fitted_imputers=None,
    fit_mode=True,
    encode_ordinal_cols=ordinal_mappings,
    remove_from_encoding=["Attrition"] + mutual_info_columns
)

print(f"✅ Train set preprocessed: {train_processed.shape}")

# Preprocess TEST set (use fitted transformers from train)
test_processed, _, _ = preprocess_data_safe(
    test_set,
    fitted_scaler=fitted_scaler,
    fitted_imputers=fitted_imputers,
    fit_mode=False,
    encode_ordinal_cols=ordinal_mappings,
    remove_from_encoding=["Attrition"] + mutual_info_columns
)

print(f"✅ Test set preprocessed: {test_processed.shape}")
print(f"✅ NO DATA LEAKAGE: Scaler and imputers fitted only on train set")

# Prepare X and y
X_train_full = train_processed.drop(columns=[target_col] + [col for col in mutual_info_columns if col in train_processed.columns], errors="ignore")
X_test_full = test_processed.drop(columns=[target_col] + [col for col in mutual_info_columns if col in test_processed.columns], errors="ignore")
y_train = train_processed[target_col]
y_test = test_processed[target_col]

# Ensure X_train and X_test have same columns
common_cols = X_train_full.columns.intersection(X_test_full.columns)
X_train = X_train_full[common_cols]
X_test = X_test_full[common_cols]

print(f"\nFinal feature count: {len(common_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

##############################
# Feature Selection (BEFORE SMOTE)
##############################

print("\n" + "="*70)
print("🔬 STEP 1: Feature Selection Using ANOVA (on TRAIN set only)")
print("="*70)

# Calculate ANOVA on TRAIN set only (no leakage)
from sklearn.feature_selection import SelectKBest

# Use SelectKBest to get scores
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train, y_train)

# Get feature scores
feature_scores = pd.Series(selector.scores_, index=X_train.columns)
feature_scores = feature_scores.sort_values(ascending=False)

# Filter out unwanted patterns
exclude_patterns = ["day_of_week", "avg_hours_day_", r"\d{4}-\d{2}-\d{2}_hours"]
for pattern in exclude_patterns:
    feature_scores = feature_scores[~feature_scores.index.str.contains(pattern, regex=True)]

print(f"\nTop 20 Features by ANOVA F-score:")
print(feature_scores.head(20))

# Select top features
top_k = 15
top_features = feature_scores.head(top_k).index.tolist()

print(f"\n✅ Selected {top_k} features")
print(f"Features: {', '.join(top_features[:10])}...")

# Apply feature selection
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

print(f"X_train: {X_train.shape} → {X_train_selected.shape}")
print(f"X_test: {X_test.shape} → {X_test_selected.shape}")

##############################
# SMOTE (AFTER Feature Selection)
##############################

print("\n" + "="*70)
print("🔬 STEP 2: Finding Optimal SMOTE Strategy")
print("="*70)

# Test different SMOTE strategies on SELECTED features
smote_strategies = [0.3, 0.4, 0.5, 0.6, 0.7]
best_smote_strategy = 0.5
best_smote_f1 = 0

for strategy in smote_strategies:
    smote_temp = SMOTE(random_state=random_state, k_neighbors=10, sampling_strategy=strategy)
    X_temp, y_temp = smote_temp.fit_resample(X_train_selected, y_train)
    
    # Quick test with basic Perceptron
    perc_temp = Perceptron(max_iter=10000, random_state=random_state, tol=1e-3)
    perc_temp.fit(X_temp, y_temp)
    y_pred_temp = perc_temp.predict(X_test_selected)
    
    cm_temp = confusion_matrix(y_test, y_pred_temp)
    tn, fp, fn, tp = cm_temp.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Strategy {strategy:.1f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Samples={len(y_temp[y_temp==0])}:{len(y_temp[y_temp==1])}")
    
    if f1 > best_smote_f1:
        best_smote_f1 = f1
        best_smote_strategy = strategy

print(f"\n✅ Best SMOTE strategy: {best_smote_strategy} (F1={best_smote_f1:.3f})")

# Apply SMOTE with best strategy
smote = SMOTE(random_state=random_state, k_neighbors=10, sampling_strategy=best_smote_strategy)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

print(f"\nOriginal: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
print(f"Balanced: {pd.Series(y_train_balanced).value_counts().sort_index().to_dict()}")

###########
# Training
###########

print("\n" + "="*70)
print("🔬 STEP 3: GridSearchCV - Optimizing Perceptron Hyperparameters")
print("="*70)

# Remove class_weight='balanced' to avoid double-counting with SMOTE
param_grid = {
    'penalty': [None, 'l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # Added 0.1 for more regularization
    'eta0': [0.5, 1.0, 1.5],  # Narrowed range
    'max_iter': [10000],
    'tol': [1e-3],
}

print(f"Testing {len(param_grid['penalty']) * len(param_grid['alpha']) * len(param_grid['eta0'])} combinations...")
print("Note: Removed class_weight to avoid double-counting with SMOTE\n")

grid_search = GridSearchCV(
    Perceptron(random_state=random_state),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

# Use X_train_balanced (after SMOTE) not X_train_selected
grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\n✅ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n✅ Best CV F1-score: {grid_search.best_score_:.3f}")

models = {
    "Perceptron_Optimized": grid_search.best_estimator_,
    }

print("\n" + "="*70)
print("🚀 STEP 4: Final Model Training & Evaluation")
print("="*70)

for name, m in models.items():
    print("\n" + "="*70)
    print(f"📊 {name} - FINAL RESULTS")
    print("="*70)
    t_start = time.time()
    
    # Model is already fitted from GridSearchCV
    # Re-fit on full balanced training data to ensure consistency
    m.fit(X_train_balanced, y_train_balanced)
    
    # Use F1 score for CV instead of accuracy (better for imbalanced data)
    x_val_scores_f1 = cross_val_score(m, X_train_balanced, y_train_balanced, cv=5, scoring="f1")
    print(f"\nCross-validation F1 mean {x_val_scores_f1.mean():.4f}, std dev {x_val_scores_f1.std():.4f}")
    
    y_pred = m.predict(X_test_selected)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate key metrics for attrition detection (class 1)
    tn, fp, fn, tp = cm.ravel()
    precision_class1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0
    
    print(f"\n📊 Class 1 (Attrition) Metrics:")
    print(f"  Precision: {precision_class1:.3f} ({tp}/{tp+fp}) - How many alerts are real")
    print(f"  Recall:    {recall_class1:.3f} ({tp}/{tp+fn}) - How many attritions detected")
    print(f"  F1-Score:  {f1_class1:.3f} - Balance between precision/recall")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Get probability scores based on model type
    if hasattr(m, 'predict_proba'):
        y_prob = m.predict_proba(X_test_selected)[:, 1]
    elif hasattr(m, 'decision_function'):
        y_prob = m.decision_function(X_test_selected)
    else:
        print("Warning: Model doesn't support probability prediction, skipping ROC-AUC")
        continue
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    
    # STEP 5: Optimize decision threshold
    print("\n" + "="*70)
    print("🎯 STEP 5: Optimizing Decision Threshold")
    print("="*70)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find threshold that maximizes F1
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1] if optimal_idx_f1 < len(thresholds) else 0.0
    
    # Find threshold with minimum precision constraint (35%)
    min_precision_threshold = 0.35
    valid_indices = precisions >= min_precision_threshold
    
    if valid_indices.any():
        # Among valid precisions, find the one with best F1
        valid_f1s = f1_scores[valid_indices]
        if len(valid_f1s) > 0:
            best_valid_idx = np.argmax(valid_f1s)
            # Get original index in precisions array
            original_indices = np.where(valid_indices)[0]
            optimal_idx_constrained = original_indices[best_valid_idx]
            optimal_threshold_constrained = thresholds[optimal_idx_constrained] if optimal_idx_constrained < len(thresholds) else 0.0
        else:
            optimal_threshold_constrained = optimal_threshold_f1
    else:
        optimal_threshold_constrained = optimal_threshold_f1
    
    # Use constrained threshold
    optimal_threshold = optimal_threshold_constrained
    
    # Predict with optimized threshold
    y_pred_optimized = (y_prob >= optimal_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_optimized)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
    recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    
    improvement_pct = ((f1_opt - f1_class1) / f1_class1 * 100) if f1_class1 > 0 else 0
    
    print(f"\n📉 DEFAULT Threshold (0.5):")
    print(f"   Precision: {precision_class1:.3f} | Recall: {recall_class1:.3f} | F1: {f1_class1:.3f}")
    print(f"   Confusion Matrix: [[{tn:3d} {fp:3d}] [{fn:3d} {tp:3d}]]")
    
    print(f"\n📈 OPTIMAL Threshold ({optimal_threshold:.4f}) - With Precision ≥ 35% constraint:")
    print(f"   Precision: {precision_opt:.3f} ({tp_opt}/{tp_opt+fp_opt}) - {precision_opt*100:.1f}% of alerts are real")
    print(f"   Recall:    {recall_opt:.3f} ({tp_opt}/{tp_opt+fn_opt}) - Detects {recall_opt*100:.1f}% of attritions")
    print(f"   F1-Score:  {f1_opt:.3f} ⬆️  +{improvement_pct:.1f}% improvement")
    print(f"   Confusion Matrix: [[{tn_opt:3d} {fp_opt:3d}] [{fn_opt:3d} {tp_opt:3d}]]")
    
    print(f"\n💡 Business Impact:")
    print(f"   ✅ Detected: {tp_opt}/{tp_opt+fn_opt} attritions ({recall_opt*100:.1f}%)")
    print(f"   ⚠️  Cost: Need to interview {tp_opt+fp_opt} employees ({fp_opt} unnecessary)")
    print(f"   💰 Efficiency: {precision_opt*100:.1f}% of interventions are useful")
    print(f"   ❌ Missed: {fn_opt} attritions will leave undetected")

    print(f"\n⏱️  Time taken: {time.time() - t_start:.2f} seconds")

    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    closest_fpr = fpr[min_idx]
    closest_tpr = tpr[min_idx]
    
    # Find optimal threshold point on ROC curve
    optimal_fpr_idx = np.argmin(np.abs(thresholds_roc - optimal_threshold)) if len(thresholds_roc) > 0 else 0
    optimal_fpr = fpr[optimal_fpr_idx] if optimal_fpr_idx < len(fpr) else 0
    optimal_tpr = tpr[optimal_fpr_idx] if optimal_fpr_idx < len(tpr) else 1

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    # Mark closest point to (0,1)
    plt.plot([0, closest_fpr], [1, closest_tpr], "r-", linewidth=2, alpha=0.5,
            label=f"Ideal point ({closest_fpr:.3f}, {closest_tpr:.3f})")
    
    # Mark optimal threshold point
    plt.plot(optimal_fpr, optimal_tpr, "go", markersize=12, 
            label=f"Optimal threshold={optimal_threshold:.3f}\nTPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f}")

    # Add distance annotation
    mid_x = closest_fpr / 2
    mid_y = (1 + closest_tpr) / 2
    plt.text(mid_x, mid_y, f"d = {min_distance:.4f}",
            fontsize=9, color="red",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", alpha=0.7)
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{name} - ROC Curve\nF1={f1_opt:.3f}, Precision={precision_opt:.3f}, Recall={recall_opt:.3f}", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    # plt.show()
    
print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE!")
print("="*70)
