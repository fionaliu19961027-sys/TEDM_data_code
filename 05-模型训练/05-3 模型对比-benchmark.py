# -*- coding: utf-8 -*-
"""
对比实验（升级版）：同时评估训练集与测试集
输出指标：MSE, RMSE, MAE, R2
结果汇总并输出到 Excel：benchmark_results_with_splits.xlsx
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

# -----------------------------
# 1) 数据加载
# -----------------------------
df = pd.read_excel("test_processed.xlsx")

X = df[[
    "Price","Flavor","Drinks","Service Attitude",
    "Atmosphere","Waitstaff","Environment","Seating",
    "len_review_text","num_punctuation","num_senti_words","total_attributes_found"
]]
y = df["star_rate"]

X = X.fillna(0)
y = pd.to_numeric(y, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# 2) 评估工具
# -----------------------------
records = []

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def evaluate_split(name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    mse,trmse,mae,r2 = _metrics(y_true_train, y_pred_train)
    records.append({"Model": name, "Split":"Train", "MSE": mse, "RMSE": trmse, "MAE": mae, "R2": r2})
    print(f"\n{name}  [Train]")
    print(f"MSE: {mse:.4f} | RMSE: {trmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    mse,rmse,mae,r2 = _metrics(y_true_test, y_pred_test)
    records.append({"Model": name, "Split":"Test", "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"{name}  [Test ]")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# -----------------------------
# 3) 训练对比模型 + 同时评估 Train/Test
# -----------------------------

# 3.1 MLR 多元线性回归
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
evaluate_split(
    "MLR",
    y_train, mlr.predict(X_train_scaled),
    y_test,  mlr.predict(X_test_scaled)
)

# 3.2 RG1 对数线性 (log1p(y) ~ X) —— 预测回到原标尺
y_train_log = np.log1p(y_train)
mlr_rg1 = LinearRegression()
mlr_rg1.fit(X_train_scaled, y_train_log)
y_pred_train_rg1 = np.expm1(mlr_rg1.predict(X_train_scaled))
y_pred_test_rg1  = np.expm1(mlr_rg1.predict(X_test_scaled))
evaluate_split(
    "RG1 (对数线性)",
    y_train, y_pred_train_rg1,
    y_test,  y_pred_test_rg1
)

# 3.3 RG2 倒数线性 (1/y ~ X) —— 预测回到原标尺
eps = 1e-6
y_train_inv = 1.0 / (y_train + eps)
mlr_rg2 = LinearRegression()
mlr_rg2.fit(X_train_scaled, y_train_inv)
y_pred_train_rg2 = 1.0 / (mlr_rg2.predict(X_train_scaled) + eps)
y_pred_test_rg2  = 1.0 / (mlr_rg2.predict(X_test_scaled)  + eps)
evaluate_split(
    "RG2 (倒数线性)",
    y_train, y_pred_train_rg2,
    y_test,  y_pred_test_rg2
)

# 3.4 PR1 多项式回归 (2阶)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly2.fit_transform(X_train_scaled)
X_test_poly2  = poly2.transform(X_test_scaled)
pr1 = LinearRegression()
pr1.fit(X_train_poly2, y_train)
evaluate_split(
    "PR1 (多项式2阶)",
    y_train, pr1.predict(X_train_poly2),
    y_test,  pr1.predict(X_test_poly2)
)

# 3.5 PR2 多项式回归 (3阶)
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train_scaled)
X_test_poly3  = poly3.transform(X_test_scaled)
pr2 = LinearRegression()
pr2.fit(X_train_poly3, y_train)
evaluate_split(
    "PR2 (多项式3阶)",
    y_train, pr2.predict(X_train_poly3),
    y_test,  pr2.predict(X_test_poly3)
)

# 3.6 BPNN (MLP)
bpnn = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=42)
bpnn.fit(X_train_scaled, y_train)
evaluate_split(
    "BPNN (MLP)",
    y_train, bpnn.predict(X_train_scaled),
    y_test,  bpnn.predict(X_test_scaled)
)

# 3.7 GRNN (近似实现: KernelRidge with RBF)
grnn = KernelRidge(kernel="rbf", gamma=0.1)
grnn.fit(X_train_scaled, y_train)
evaluate_split(
    "GRNN (近似:KernelRidge-RBF)",
    y_train, grnn.predict(X_train_scaled),
    y_test,  grnn.predict(X_test_scaled)
)

# 3.8 严格RBFNN (用 SVR-RBF 模拟，较严格参数)
rbf_strict = SVR(kernel="rbf", C=1.0, gamma=0.1)
rbf_strict.fit(X_train_scaled, y_train)
evaluate_split(
    "严格 RBFNN (SVR-RBF)",
    y_train, rbf_strict.predict(X_train_scaled),
    y_test,  rbf_strict.predict(X_test_scaled)
)

# 3.9 近似RBFNN (SVR-RBF 宽松参数)
rbf_approx = SVR(kernel="rbf", C=0.5, gamma=0.05)
rbf_approx.fit(X_train_scaled, y_train)
evaluate_split(
    "近似 RBFNN (SVR-RBF)",
    y_train, rbf_approx.predict(X_train_scaled),
    y_test,  rbf_approx.predict(X_test_scaled)
)

# 3.10 SVM (线性核)
svm_linear = SVR(kernel="linear", C=1.0)
svm_linear.fit(X_train_scaled, y_train)
evaluate_split(
    "SVM (线性核)",
    y_train, svm_linear.predict(X_train_scaled),
    y_test,  svm_linear.predict(X_test_scaled)
)

# -----------------------------
# 4) 汇总导出
# -----------------------------
df_out = pd.DataFrame(records)
# 排序更直观：按模型名、Split顺序（Train在前、Test在后）
split_cat = pd.Categorical(df_out["Split"], categories=["Train","Test"], ordered=True)
df_out = df_out.assign(Split=split_cat).sort_values(["Model","Split"]).reset_index(drop=True)

df_out.to_excel("benchmark_results_with_splits.xlsx", index=False)
print("\n=== 结果已保存到 benchmark_results_with_splits.xlsx ===")
print(df_out)
