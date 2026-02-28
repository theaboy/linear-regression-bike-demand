
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!pip install numpy
!git clone https://github.com/theaboy/linear-regression-bike-demand.git

"""#Import Data and Clean Up"""

df = pd.read_csv("linear-regression-bike-demand/day.csv")
display(df.head())
df.shape
df.info()
n_rows, n_columns = np.shape(df)
print(f"# Instances / data points/ rows / tuples (rows): {n_rows}")
print(f"# Columns / features / attributes: {n_columns}")

# ===============================
# Data validation & outlier capping
# ===============================

df = df.copy()

# 1) Parse date safely
df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

# 2) Define validation rules
rules = {
    "instant":      ("int",   (1, None)),
    "season":       ("cat",   {1,2,3,4}),
    "yr":           ("cat",   {0,1}),
    "mnth":         ("int",   (1,12)),
    "holiday":      ("cat",   {0,1}),
    "weekday":      ("cat",   {0,1,2,3,4,5,6}),
    "workingday":   ("cat",   {0,1}),
    "weathersit":   ("cat",   {1,2,3,4}),
    "temp":         ("float", (0,1)),
    "atemp":        ("float", (0,1)),
    "hum":          ("float", (0,1)),
    "windspeed":    ("float", (0,1)),
    "casual":       ("int",   (0, None)),
    "registered":   ("int",   (0, None)),
    "cnt":          ("int",   (0, None)),
}

# 3) Run validation checks
malformed_report = {}

for col, (kind, rule) in rules.items():
    if col not in df.columns:
        continue

    s = df[col]

    # Coerce numeric types
    if kind in ["int", "float"] and s.dtype == "object":
        df[col] = pd.to_numeric(s, errors="coerce")
        s = df[col]

    if kind == "cat":
        bad = ~s.isin(rule) & s.notna()
        malformed_report[col] = int(bad.sum())

    else:  # numeric
        lo, hi = rule
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= (s < lo)
        if hi is not None:
            bad |= (s > hi)
        bad &= s.notna()
        malformed_report[col] = int(bad.sum())

print("\n=== MALFORMED VALUE SUMMARY ===")
print(pd.Series(malformed_report).sort_values(ascending=False))

# 4) Date checks
bad_dates = df["dteday"].isna().sum()
dup_days = df["dteday"].duplicated().sum()

print(f"\nBad / unparseable dates: {bad_dates}")
print(f"Duplicate dates: {dup_days}")

# 5) IQR outlier detection
num_cols = df.select_dtypes(include=[np.number]).columns
outlier_report = {}

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier_report[col] = ((df[col] < low) | (df[col] > high)).sum()

print("\n=== OUTLIER COUNTS (IQR rule) ===")
print(pd.Series(outlier_report).sort_values(ascending=False))

# 6) Cap continuous features
continuous_cols = ["temp", "atemp", "hum", "windspeed"]

for col in continuous_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df[col] = df[col].clip(low, high) #bring to bounds

print("\n Outliers capped for:", continuous_cols)

"""#Continuous Feature Exploration"""

features_to_check = ["temp", "atemp", "hum", "windspeed"]

for col in features_to_check:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df["cnt"], alpha=0.6)

    # ---- Trend line (linear) ----
    x = df[col].values
    y = df["cnt"].values

    z1 = np.polyfit(x, y, 1)          # degree 1
    p1 = np.poly1d(z1)

    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, p1(xs), linewidth=2, label="Linear trend")

    # ---- Trend line (quadratic) ----
    z2 = np.polyfit(x, y, 2)          # degree 2
    p2 = np.poly1d(z2)

    plt.plot(xs, p2(xs), linewidth=2, label="Quadratic trend")

    plt.title(f"cnt vs {col} (with trend lines)")
    plt.xlabel(col)
    plt.ylabel("cnt")
    plt.legend()
    plt.show()

"""#Discrete Feature Exploration"""

cat_cols = ["season", "weathersit", "weekday", "workingday", "holiday", "mnth", "yr"]

for col in cat_cols:
    df.boxplot(column="cnt", by=col, figsize=(6,4))
    plt.title(f"cnt by {col}")
    plt.suptitle("")  # removes automatic pandas title
    plt.xlabel(col)
    plt.ylabel("cnt")
    plt.show()

"""#Preparing Design and Target Tables"""

columns_to_drop = ['instant', 'casual', 'registered', 'holiday', 'cnt','dteday']
df_design = df.drop(columns=columns_to_drop)
df_cnt = df[['cnt']]

print(f'Columns removed: {columns_to_drop}')
print('\nDataFrame after dropping columns:')
display(df_design.head())
display(df_cnt.head())

print(df_design.info())
print(df_cnt.info())

categorical_cols = ["season", "weathersit", "weekday", "mnth"]

df_design = pd.get_dummies(
    df_design,
    columns = categorical_cols,  # avoids dummy variable trap
)
df_design = df_design.apply(pd.to_numeric, errors="raise")

design = df_design.to_numpy(dtype=np.float64)
feature_columns = list(df_design.columns)

#add y-intercept to the design matrix

design = np.column_stack((np.ones(design.shape[0]), design))
cnt = df_cnt.to_numpy()
print(df_design.info())

"""#Seperating Data into Test and Trainning Sets"""

np.random.seed(42)

n = design.shape[0]
indices = np.random.permutation(n)

train_size = int(0.8 * n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = design[train_idx]
y_train = cnt[train_idx]

X_test = design[test_idx]
y_test = cnt[test_idx]
# ---- NORMALIZATION (from scratch) ----

# Copy to avoid modifying original arrays
X_train_norm = X_train.copy()
X_test_norm  = X_test.copy()

# Compute mean and std from TRAINING data ONLY (excluding bias column)
mean = X_train_norm[:, 1:].mean(axis=0)
std  = X_train_norm[:, 1:].std(axis=0)

# Avoid division by zero
std[std == 0] = 1

# Normalize (exclude bias column at index 0)
X_train_norm[:, 1:] = (X_train_norm[:, 1:] - mean) / std
X_test_norm[:, 1:]  = (X_test_norm[:, 1:]  - mean) / std




Beta = np.linalg.lstsq(X_train_norm, y_train, rcond=None)[0]
# w, residuals, rank, singular_values

display(Beta)

"""#Linear Regression"""

y_train_pred = X_train_norm @ Beta
y_test_pred = X_test_norm @ Beta

"""#Calculate Mean Squared Error"""

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)

"""#Results Visualization"""

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red"
)
plt.xlabel("True bike count")
plt.ylabel("Predicted bike count")
plt.title("Predicted vs True Bike Demand (Test Set)")
plt.show()

residuals = y_test - y_test_pred

plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--", color="red")
plt.xlabel("Predicted bike count")
plt.ylabel("Residual (true − predicted)")
plt.title("Residuals vs Predicted Values")
plt.show()

"""# Most influencal features"""

# ===============================
# Standardize continuous features
# ===============================

feature_names = ["intercept"] + list(df_design.columns)

continuous_cols = ["temp", "atemp", "hum", "windspeed"]
continuous_idx = [feature_names.index(col) for col in continuous_cols]

X_train_std = X_train.copy()
X_test_std = X_test.copy()

mean = X_train[:, continuous_idx].mean(axis=0)
std = X_train[:, continuous_idx].std(axis=0)
std[std == 0] = 1

X_train_std[:, continuous_idx] = (X_train[:, continuous_idx] - mean) / std
X_test_std[:, continuous_idx]  = (X_test[:, continuous_idx]  - mean) / std

# ==========================
# Most influencal features
# =========================
print("y_train mean:", float(y_train.mean()))
print("y_train std :", float(y_train.std()))
Beta_std = np.linalg.lstsq(X_train_std, y_train, rcond=None)[0]
coef_table = pd.DataFrame({
    "feature": feature_names,
    "beta": Beta_std.flatten(),
    "abs_beta": np.abs(Beta_std.flatten())
}).sort_values("abs_beta", ascending=False)

coef_table_no_intercept = coef_table[coef_table.feature != "intercept"]
display(coef_table_no_intercept)

# ============================================
# Train a model using ONLY the most influential features
# ============================================

# Choose how many top features you want (change this) ps : to run you should RUN ALL not individually run
top_k = 10


coef_table_no_intercept = coef_table[coef_table["feature"] != "intercept"].copy()

top_features = (
    coef_table_no_intercept
    .sort_values("abs_beta", ascending=False)
    .head(top_k)["feature"]
    .tolist()
)

print("Top features:", top_features)


feature_names = ["intercept"] + list(df_design.columns)

top_idx = [feature_names.index(f) for f in top_features]


X_train_top = X_train_std[:, [0] + top_idx]
X_test_top  = X_test_std[:,  [0] + top_idx]


Beta_top = np.linalg.lstsq(X_train_top, y_train, rcond=None)[0]


y_pred_train = X_train_top @ Beta_top
y_pred_test  = X_test_top  @ Beta_top


mse_train = np.mean((y_train - y_pred_train) ** 2)
mse_test  = np.mean((y_test  - y_pred_test)  ** 2)



print(f"Train MSE: {float(mse_train):.2f}")
print(f"Test  MSE: {float(mse_test):.2f}")


coef_top = pd.DataFrame({
    "feature": ["intercept"] + top_features,
    "beta": Beta_top.flatten(),
    "abs_beta": np.abs(Beta_top.flatten())
}).sort_values("abs_beta", ascending=False)

display(coef_top)

"""#Feature Engineering"""

def add_polynomial_features(X, degree=2):
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly

def add_interaction_terms(X):
    n_samples, n_features = X.shape
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack([X] + interactions)

X_train_base = X_train_norm[:, 1:]
X_test_base  = X_test_norm[:, 1:]

X_train_fe = add_interaction_terms(add_polynomial_features(X_train_base, 2))
X_test_fe  = add_interaction_terms(add_polynomial_features(X_test_base, 2))


mean = X_train_fe.mean(axis=0)
std = X_train_fe.std(axis=0) + 1e-8

X_train_fe = (X_train_fe - mean) / std
X_test_fe  = (X_test_fe - mean) / std

X_train_fe = np.hstack([np.ones((X_train_fe.shape[0], 1)), X_train_fe])
X_test_fe  = np.hstack([np.ones((X_test_fe.shape[0], 1)), X_test_fe])

theta = np.linalg.pinv(X_train_fe.T @ X_train_fe) @ X_train_fe.T @ y_train


y_pred_train = X_train_fe @ theta
y_pred_test  = X_test_fe @ theta

mse_train = np.mean((y_train - y_pred_train) ** 2)
mse_test  = np.mean((y_test - y_pred_test) ** 2)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linewidth=2)
plt.xlabel("Actual cnt")
plt.ylabel("Predicted cnt")
plt.title("Actual vs Predicted (Feature Engineered Model)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred_test

plt.figure(figsize=(7,4))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(0, linewidth=2)
plt.xlabel("Predicted cnt")
plt.ylabel("Residuals")
plt.title("Residual Plot (Feature Engineered Model)")
plt.grid(True)
plt.show()

import numpy as np

# indices of features
temp_idx  = feature_columns.index("temp")
atemp_idx = feature_columns.index("atemp")
hum_idx   = feature_columns.index("hum")
wind_idx  = feature_columns.index("windspeed")

# ---- FEATURE ENGINEERING ON SCALED DATA ----
X_train_fe = np.column_stack([
    X_train,                                # scaled original features
    X_train[:, temp_idx]**2,                # temp^2
    X_train[:, atemp_idx]**2,               # atemp^2
    X_train[:, wind_idx]**2,                # windspeed^2
    X_train[:, temp_idx] * X_train[:, hum_idx]  # temp × hum
])

X_test_fe = np.column_stack([
    X_test,
    X_test[:, temp_idx]**2,
    X_test[:, atemp_idx]**2,
    X_test[:, wind_idx]**2,
    X_test[:, temp_idx] * X_test[:, hum_idx]
])

# ---- ADD BIAS (ONCE) ----
X_train_fe = np.c_[np.ones(X_train_fe.shape[0]), X_train_fe]
X_test_fe  = np.c_[np.ones(X_test_fe.shape[0]),  X_test_fe]

# ---- TRAIN LINEAR REGRESSION ----
theta = np.linalg.pinv(X_train_fe.T @ X_train_fe) @ (X_train_fe.T @ y_train)

# ---- PREDICT ----
y_pred_train = X_train_fe @ theta
y_pred_test  = X_test_fe @ theta

# ---- EVALUATE ----
mse_train = np.mean((y_train - y_pred_train)**2)
mse_test  = np.mean((y_test - y_pred_test)**2)

print("Train MSE:", mse_train)
print("Test  MSE:", mse_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linewidth=2)
plt.xlabel("Actual cnt")
plt.ylabel("Predicted cnt")
plt.title("Actual vs Predicted (Feature Engineered Model)")
plt.grid(True)
plt.show()

residuals = y_test - y_pred_test

plt.figure(figsize=(7,4))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(0, linewidth=2)
plt.xlabel("Predicted cnt")
plt.ylabel("Residuals")
plt.title("Residual Plot (Feature Engineered Model)")
plt.grid(True)
plt.show()

"""#Perform Same Linear Regression Process for Casual vs Registered Bike Rental Ratio"""

# ===============================
# Data validation & outlier capping
# ===============================

df = df.copy()

# 1) Parse date safely
df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

# Create the new target variable 'casual_registered_ratio'
df['casual_registered_ratio'] = df['casual'] / (df['registered'])
# df'casual_registered_ratio' as the target variable
df_ratio = df[['casual_registered_ratio']]

# 2) Define validation rules
rules = {
    "instant":      ("int",   (1, None)),
    "season":       ("cat",   {1,2,3,4}),
    "yr":           ("cat",   {0,1}),
    "mnth":         ("int",   (1,12)),
    "holiday":      ("cat",   {0,1}),
    "weekday":      ("cat",   {0,1,2,3,4,5,6}),
    "workingday":   ("cat",   {0,1}),
    "weathersit":   ("cat",   {1,2,3,4}),
    "temp":         ("float", (0,1)),
    "atemp":        ("float", (0,1)),
    "hum":          ("float", (0,1)),
    "windspeed":    ("float", (0,1)),
    "casual":       ("int",   (0, None)),
    "registered":   ("int",   (0, None)),
    "cnt":          ("int",   (0, None)),
    "casual_registered_ratio": ("float", (0, None))
    }

# 3) Run validation checks
malformed_report = {}

for col, (kind, rule) in rules.items():
    if col not in df.columns:
        continue

    s = df[col]

    # Coerce numeric types
    if kind in ["int", "float"] and s.dtype == "object":
        df[col] = pd.to_numeric(s, errors="coerce")
        s = df[col]

    if kind == "cat":
        bad = ~s.isin(rule) & s.notna()
        malformed_report[col] = int(bad.sum())

    else:  # numeric
        lo, hi = rule
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= (s < lo)
        if hi is not None:
            bad |= (s > hi)
        bad &= s.notna()
        malformed_report[col] = int(bad.sum())

print("\n=== MALFORMED VALUE SUMMARY ===")
print(pd.Series(malformed_report).sort_values(ascending=False))

# 4) Date checks
bad_dates = df["dteday"].isna().sum()
dup_days = df["dteday"].duplicated().sum()

print(f"\nBad / unparseable dates: {bad_dates}")
print(f"Duplicate dates: {dup_days}")

# 5) IQR outlier detection
num_cols = df.select_dtypes(include=[np.number]).columns
outlier_report = {}

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier_report[col] = ((df[col] < low) | (df[col] > high)).sum()

print("\n=== OUTLIER COUNTS (IQR rule) ===")
print(pd.Series(outlier_report).sort_values(ascending=False))

# 6) Cap continuous features
continuous_cols = ["temp", "atemp", "hum", "windspeed"]

for col in continuous_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df[col] = df[col].clip(low, high) #bring to bounds

print("\n Outliers capped for:", continuous_cols)

"""#Continuous Feature Exploration"""

features_to_check = ["temp", "atemp", "hum", "windspeed"]

for col in features_to_check:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df_ratio["casual_registered_ratio"], alpha=0.6)

    # ---- Trend line (linear) ----
    x = df[col].values
    y = df["casual_registered_ratio"].values

    z1 = np.polyfit(x, y, 1)          # degree 1
    p1 = np.poly1d(z1)

    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, p1(xs), linewidth=2, label="Linear trend")

    # ---- Trend line (quadratic) ----
    z2 = np.polyfit(x, y, 2)          # degree 2
    p2 = np.poly1d(z2)

    plt.plot(xs, p2(xs), linewidth=2, label="Quadratic trend")

    plt.title(f"ratio vs {col} (with trend lines)")
    plt.xlabel(col)
    plt.ylabel("ratio")
    plt.legend()
    plt.show()

"""#Discrete Feature Exploration"""

cat_cols = ["season", "weathersit", "weekday", "workingday", "holiday", "mnth", "yr"]

for col in cat_cols:
    df.boxplot(column="casual_registered_ratio", by=col, figsize=(6,4))
    plt.title(f"ratio by {col}")
    plt.suptitle("")  # removes automatic pandas title
    plt.xlabel(col)
    plt.ylabel("ratio")
    plt.show()

"""#Prepare Design and Target Tables"""

columns_to_drop = ['instant', 'casual', 'registered', 'holiday', 'yr', 'weathersit', 'season', 'cnt','dteday', 'casual_registered_ratio']
df_design = df.drop(columns=columns_to_drop)

print(f'Columns removed for df_design: {columns_to_drop}')
print('\nDataFrame after dropping columns (df_design):')
display(df_design.head())
print('\nDataFrame containing the new target variable (df_ratio):')
display(df_ratio.head())

print(df_design.info())
print(df_ratio.info())

#Conver to numpy arrays

feature_columns = list(df_design.columns)
design = df_design.to_numpy()
#add y-intercept to the design matrix
design = np.column_stack((np.ones(design.shape[0]), design))
ratio = df_ratio.to_numpy()

"""#Seperating Data into Test and Trainning Sets"""

np.random.seed(42)

n = design.shape[0]
indices = np.random.permutation(n)   #Insuring a random seperation between both sets

train_size = int(0.8 * n)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = design[train_idx]
y_train = ratio[train_idx]

X_test = design[test_idx]
y_test = ratio[test_idx]

Beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
# w, residuals, rank, singular_values

display(Beta)

"""#Linear Regression"""

y_train_pred = X_train @ Beta
y_test_pred = X_test @ Beta

"""#Calculate Mean Squared Error"""

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)

"""#Results Visualization"""

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red"
)
plt.xlabel("True casual vs registered bikes ratio")
plt.ylabel("Predicted casual vs registered bikes ratio")
plt.title("Predicted vs True Casual/Registered Bike Rental Demand (Test Set)")
plt.show()

residuals = y_test - y_test_pred

plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--", color="red")
plt.xlabel("Predicted bike rental ratio")
plt.ylabel("Residual (true − predicted)")
plt.title("Residuals vs Predicted Values")
plt.show()
