# -*- coding: utf-8 -*-
"""
Two-stage consumer rating model with residual correction
- NN with Huber loss
- Residual correction with CatBoost
- Stage-2 gating: if veto, skip compensation (S_total = S_base)
- Print NN parameters and Boost feature importances

K=5 交叉验证版：仅将原先的 train_test_split 改为 5 折 KFold，其余不变。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from catboost import CatBoostRegressor   # ✅ CatBoost 残差修正

# -----------------------------
# 0) Reproducibility & device
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1) Data loading
# -----------------------------
DATA_DIR = Path(".")
df_reviews = pd.read_excel(DATA_DIR / "test_processed.xlsx")
df_prefs   = pd.read_excel(DATA_DIR / "consumer_preference_matrix.xlsx")
df_strict  = pd.read_excel(DATA_DIR / "strictness_analysis.xlsx")
df_utility = pd.read_excel(DATA_DIR / "enhanced_utility_analysis.xlsx")

for d in (df_reviews, df_prefs, df_strict, df_utility):
    d.columns = d.columns.astype(str).str.strip()
    if "reviewer_id" in d.columns:
        d["reviewer_id"] = d["reviewer_id"].astype(str)

basic_attrs = ["Environment", "Price", "Waitstaff", "Service Attitude"]
other_attrs = ["Flavor", "Drinks", "Atmosphere", "Seating"]
all_attrs   = basic_attrs + other_attrs
control_vars = ["len_review_text", "num_punctuation", "num_senti_words", "total_attributes_found"]

# preference matrix: rename to *_w
pref_cols = [c for c in df_prefs.columns if c != "reviewer_id"]
df_prefs = df_prefs.rename(columns={c: f"{c}_w" for c in pref_cols})

# left-join to keep all reviews
df = (df_reviews
      .merge(df_prefs, on="reviewer_id", how="left")
      .merge(df_strict, on="reviewer_id", how="left")
      .merge(df_utility, on="reviewer_id", how="left"))

df["star_rate"] = pd.to_numeric(df["star_rate"], errors="coerce")
utility_map = {"乐观型": 0, "理性型": 1, "挑剔型": 2}
df["utility_type"] = df["final_utility_type"].map(utility_map)

# controls: fillna(0)
for c in control_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

needed_weight_cols = [f"{c}_w" for c in all_attrs]
needed_cols = all_attrs + needed_weight_cols + ["star_rate", "strictness_index", "utility_type"]
df = df.dropna(subset=needed_cols)

def to_float(df_, cols):
    return df_[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

X_attr = to_float(df, all_attrs)
PREFS  = to_float(df, needed_weight_cols)
CTRL   = to_float(df, control_vars)
STRICT = pd.to_numeric(df["strictness_index"], errors="coerce").to_numpy(dtype=np.float32)
UTIL_T = pd.to_numeric(df["utility_type"], errors="coerce").fillna(1).to_numpy(dtype=np.int64)
Y      = pd.to_numeric(df["star_rate"], errors="coerce").to_numpy(dtype=np.float32)

# -----------------------------
# 2) Feature engineering: Standardize + PCA for PREFS & CTRL
# -----------------------------
# 保持与原脚本一致：在全量样本上拟合 scaler/PCA（不在折内拟合）
scaler_prefs = StandardScaler()
scaler_ctrl  = StandardScaler()
PREFS_scaled = scaler_prefs.fit_transform(PREFS)
CTRL_scaled  = scaler_ctrl.fit_transform(CTRL)

pca_prefs = PCA(n_components=0.95, random_state=SEED)
pca_ctrl  = PCA(n_components=0.95, random_state=SEED)
PREFS_pca = pca_prefs.fit_transform(PREFS_scaled)
CTRL_pca  = pca_ctrl.fit_transform(CTRL_scaled)

print("PREFS PCA components:", PREFS_pca.shape[1])
print("CTRL PCA components :", CTRL_pca.shape[1])

# -----------------------------
# 3) K-Fold split (K=5)
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def tt(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=DEVICE)

# -----------------------------
# 4) Two-stage NN Model with Huber loss (与原版一致)
# -----------------------------
class TwoStageModel(nn.Module):
    def __init__(self, basic_attr_count, other_attr_count, pref_dim, ctrl_dim):
        super().__init__()
        self.basic_attr_count = basic_attr_count
        self.other_attr_count = other_attr_count

        # trainable & bounded via transforms
        self.strict_thresh_raw = nn.Parameter(torch.tensor(0.0))           # ∈[0,1]
        self.T_basic_raw       = nn.Parameter(torch.zeros(basic_attr_count))# ∈[-1,1]
        self.mu0_raw   = nn.Parameter(torch.tensor(0.0))   # ∈[1,3]
        self.kappa_raw = nn.Parameter(torch.tensor(0.0))   # ∈[0,2]
        self.eta_raw   = nn.Parameter(torch.tensor(0.0))   # ∈[2.5,4]

        self.beta_raw  = nn.Parameter(torch.zeros(other_attr_count))  # ≥0
        self.alpha_raw = nn.Parameter(torch.zeros(other_attr_count))  # >0

        self.delta_pref = nn.Linear(pref_dim, 1, bias=False)  # linear control on prefs PCA
        self.delta_ctrl = nn.Linear(ctrl_dim, 1, bias=False)  # linear control on ctrl PCA

        self.gamma0 = nn.Parameter(torch.tensor(0.0))
        self.gamma1 = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _sigmoid_bound(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _softplus_pos(raw, eps=1e-3):
        return torch.nn.functional.softplus(raw) + eps

    def forward(self, x, prefs, strictness, utility_type, ctrl):
        # bounded parameters
        strict_thresh = self._sigmoid_bound(self.strict_thresh_raw, 0.0, 1.0)
        T_basic       = self._sigmoid_bound(self.T_basic_raw, -1.0, 1.0)

        mu0   = self._sigmoid_bound(self.mu0_raw,   1.0, 3.0)
        kappa = self._sigmoid_bound(self.kappa_raw, 0.0, 2.0)
        eta   = self._sigmoid_bound(self.eta_raw,   2.5, 4.0)

        beta  = self._softplus_pos(self.beta_raw)
        alpha = self._softplus_pos(self.alpha_raw)

        # ----- Stage 1: Non-compensatory -----
        P_basic  = x[:, :self.basic_attr_count]
        T_expand = T_basic.unsqueeze(0).expand_as(P_basic)

        # Veto when any basic attribute < threshold AND strictness above threshold
        veto_mask = (P_basic < T_expand).any(dim=1) & (strictness > strict_thresh)

        # Low-score if veto: mu0 - kappa * max shortfall
        penalty_depth = torch.relu(T_expand - P_basic).max(dim=1).values
        veto_score = mu0 - kappa * penalty_depth

        # Otherwise accumulate basic utility up to eta
        base_effect = torch.relu(P_basic - T_expand).sum(dim=1)
        base_effect = torch.minimum(base_effect, eta.expand_as(base_effect))

        S_base = torch.where(veto_mask, veto_score, base_effect)

        # ----- Stage 2: Compensatory with hard gating -----
        P_other = x[:, self.basic_attr_count:]
        U_sum = torch.zeros_like(S_base)
        eps = 1e-6

        for i in range(self.other_attr_count):
            P_i = P_other[:, i]
            b_i = beta[i]
            a_i = alpha[i]
            U_lin   = b_i * P_i
            U_opt   = b_i * (torch.exp(a_i * P_i) - 1.0) / (torch.exp(a_i) - 1.0 + eps)
            U_picky = b_i * (1.0 - torch.exp(-a_i * P_i)) / (1.0 - torch.exp(-a_i) + eps)
            U_sel = torch.where(utility_type == 0, U_opt,
                     torch.where(utility_type == 1, U_lin, U_picky))
            U_sum = U_sum + U_sel

        # Hard gate: if veto -> S_total = S_base ; else -> S_base + U_sum
        S_total = torch.where(veto_mask, S_base, S_base + U_sum)

        # ----- Controls (linear terms on PCA features) -----
        ctrl_effect = self.delta_pref(prefs).squeeze() + self.delta_ctrl(ctrl).squeeze()
        S_final = S_total + ctrl_effect

        # ----- Logistic mapping to [1,5] -----
        y_pred = 1.0 + 4.0 / (1.0 + torch.exp(-(self.gamma0 + self.gamma1 * S_final)))
        return y_pred

# -----------------------------
# 5) K-Fold training & evaluation (保持超参不变)
# -----------------------------
EPOCHS = 200
LR = 1e-2
WD = 1e-4

fold_metrics = []  # 存每折 (train_mse, train_rmse, train_mae, train_r2, test_mse, test_rmse, test_mae, test_r2)

fold_id = 1
for train_idx, test_idx in kf.split(X_attr):
    print(f"\n================ Fold {fold_id}/5 ================\n")

    # 划分本折数据
    X_tr, X_te = X_attr[train_idx], X_attr[test_idx]
    PREFS_tr, PREFS_te = PREFS_pca[train_idx], PREFS_pca[test_idx]
    CTRL_tr, CTRL_te = CTRL_pca[train_idx], CTRL_pca[test_idx]
    STR_tr, STR_te = STRICT[train_idx], STRICT[test_idx]
    UTYPE_tr, UTYPE_te = UTIL_T[train_idx], UTIL_T[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    # 转 tensor
    X_tr_t     = tt(X_tr)
    X_te_t     = tt(X_te)
    PREFS_tr_t = tt(PREFS_tr)
    PREFS_te_t = tt(PREFS_te)
    CTRL_tr_t  = tt(CTRL_tr)
    CTRL_te_t  = tt(CTRL_te)
    STR_tr_t   = tt(STR_tr)
    STR_te_t   = tt(STR_te)
    UTYPE_tr_t = torch.tensor(UTYPE_tr, dtype=torch.long, device=DEVICE)
    UTYPE_te_t = torch.tensor(UTYPE_te, dtype=torch.long, device=DEVICE)
    Y_tr_t     = tt(Y_tr)
    Y_te_t     = tt(Y_te)

    # 模型 & 训练器
    torch.manual_seed(SEED)  # 每折重置随机性
    model = TwoStageModel(len(basic_attrs), len(other_attrs), PREFS_tr.shape[1], CTRL_tr.shape[1]).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # 训练
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        y_hat = model(X_tr_t, PREFS_tr_t, STR_tr_t, UTYPE_tr_t, CTRL_tr_t)
        loss = criterion(y_hat, Y_tr_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Fold {fold_id} | Epoch {epoch:03d}/{EPOCHS} | Train Loss={loss.item():.4f}")

    # -----------------------------
    # 6) Residual correction with CatBoost (每折)
    # -----------------------------
    model.eval()
    with torch.no_grad():
        y_tr_pred_nn = model(X_tr_t, PREFS_tr_t, STR_tr_t, UTYPE_tr_t, CTRL_tr_t).cpu().numpy()
        y_te_pred_nn = model(X_te_t, PREFS_te_t, STR_te_t, UTYPE_te_t, CTRL_te_t).cpu().numpy()

    residuals = Y_tr - y_tr_pred_nn
    X_train_features = np.hstack([X_tr, PREFS_tr, CTRL_tr, STR_tr.reshape(-1,1), UTYPE_tr.reshape(-1,1)])
    X_test_features  = np.hstack([X_te, PREFS_te, CTRL_te, STR_te.reshape(-1,1), UTYPE_te.reshape(-1,1)])

    print("\n>>> 使用 CatBoost 作为残差修正模块（本折）")
    cat = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=False
    )
    cat.fit(X_train_features, residuals)
    residuals_pred_train = cat.predict(X_train_features)
    residuals_pred_test  = cat.predict(X_test_features)

    y_tr_pred_final = y_tr_pred_nn + residuals_pred_train
    y_te_pred_final = y_te_pred_nn + residuals_pred_test

    # -----------------------------
    # 7) Evaluation (本折 + 累计)
    # -----------------------------
    def report(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n=== {name} ===")
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("MAE :", mae)
        print("R2  :", r2)
        return mse, rmse, mae, r2

    tr_mse, tr_rmse, tr_mae, tr_r2 = report(f"训练集 (NN+CatBoost) | Fold {fold_id}", Y_tr, y_tr_pred_final)
    te_mse, te_rmse, te_mae, te_r2 = report(f"验证集 (NN+CatBoost) | Fold {fold_id}", Y_te, y_te_pred_final)

    fold_metrics.append((tr_mse, tr_rmse, tr_mae, tr_r2, te_mse, te_rmse, te_mae, te_r2))

    # -----------------------------
    # 8) Print learned parameters (每折打印，保持原习惯)
    # -----------------------------
    print("\n=== NN 部分学习到的参数（Fold {}）===".format(fold_id))
    with torch.no_grad():
        print("strict_thresh (严格阈值 ∈[0,1]):", model._sigmoid_bound(model.strict_thresh_raw, 0, 1).item())
        print("T_basic (基础属性阈值 ∈[-1,1]):", model._sigmoid_bound(model.T_basic_raw, -1, 1).cpu().numpy())
        print("mu0 (否决基准分 ∈[1,3]):", model._sigmoid_bound(model.mu0_raw, 1, 3).item())
        print("kappa (踩雷惩罚 ∈[0,2]):", model._sigmoid_bound(model.kappa_raw, 0, 2).item())
        print("eta (基础效用上限 ∈[2.5,4]):", model._sigmoid_bound(model.eta_raw, 2.5, 4).item())
        print("beta (补偿属性幅度 ≥0):", model._softplus_pos(model.beta_raw).cpu().numpy())
        print("alpha (补偿属性曲率 >0):", model._softplus_pos(model.alpha_raw).cpu().numpy())
        print("gamma0 (Logistic 偏置):", model.gamma0.item())
        print("gamma1 (Logistic 斜率):", model.gamma1.item())
        print("delta_pref (偏好控制权重):", model.delta_pref.weight.detach().cpu().numpy())
        print("delta_ctrl (控制变量权重):", model.delta_ctrl.weight.detach().cpu().numpy())

    print("\n=== CatBoost 部分参数（Fold {}）===".format(fold_id))
    print("CatBoost iterations:", cat.tree_count_)
    print("Feature importances (CatBoost):")
    for fname, imp in zip(
        [f"feat_{i}" for i in range(X_train_features.shape[1])],
        cat.get_feature_importance()
    ):
        print(f"  {fname}: {imp}")

    fold_id += 1

# -----------------------------
# 9) 汇总 5 折均值与标准差
# -----------------------------
metrics_arr = np.array(fold_metrics)  # shape: (5, 8)
means = metrics_arr.mean(axis=0)
stds  = metrics_arr.std(axis=0)

print("\n================ 5-Fold Summary ================\n")
names = ["Train MSE", "Train RMSE", "Train MAE", "Train R2", "Val MSE", "Val RMSE", "Val MAE", "Val R2"]
for i, name in enumerate(names):
    print(f"{name}: {means[i]:.6f} ± {stds[i]:.6f}")


# ============================================================
# 🔚 Section: Visualization - Predicted vs Actual Rating Plot
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设你在前面的 K-fold 循环中已经定义并更新了以下两个全局列表：
# all_y_true.extend(y_val.tolist())
# all_y_pred.extend(y_pred_final.tolist())

# 如果还没定义，请在 KFold 开始前加：
# all_y_true, all_y_pred = [], []

# 转换为 NumPy 数组  Y_tr, y_tr_pred_final
y_true = np.array(Y_tr)
y_pred = np.array(y_tr_pred_final)

# 轻微抖动，避免散点挤在整数刻度竖线上
rng = np.random.default_rng(42)
jitter = rng.uniform(-0.08, 0.08, size=len(y_true))
x_jittered = y_true + jitter

# 计算每个真实星级对应的平均预测值
df = pd.DataFrame({'actual': y_true, 'pred': y_pred})
mean_pred = df.groupby('actual')['pred'].mean()

fig, ax = plt.subplots(figsize=(5.5, 4.5))

# 1) 散点：所有验证样本的预测
ax.scatter(x_jittered,
           y_pred,
           s=10,
           alpha=0.22,
           label='Validation samples')

# 2) 每个真实星级的平均预测值（折线 + 点）
ax.plot(mean_pred.index,
        mean_pred.values,
        marker='o',
        linestyle='-',
        linewidth=1.6,
        markersize=5,
        label='Mean prediction')

# 3) 理想对角线 y = x
ax.plot([1, 5],
        [1, 5],
        linestyle='--',
        linewidth=1.2,
        label='Ideal line (y = x)')

# 坐标与标签设置
ax.set_xlim(0.9, 5.1)
ax.set_ylim(0.9, 5.1)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])

ax.set_xlabel('Actual rating', fontsize=10)
ax.set_ylabel('Predicted rating', fontsize=10)
ax.set_title('TEDM: Predicted vs Actual Ratings (Validation, 5-fold CV)',
             fontsize=11, pad=8)

ax.legend(frameon=False, fontsize=9, loc='best')
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('fig_pred_vs_actual_TEDM_scatter_mean.png',
            dpi=300,
            bbox_inches='tight')
plt.show()

print("Figure saved as 'fig_pred_vs_actual_TEDM_scatter_mean.png'")


import numpy as np
import matplotlib.pyplot as plt

# 确保 all_y_true / all_y_pred 已包含所有折的验证集预测结果 Y_te, y_te_pred_final
y_true = np.array(Y_te)
y_pred = np.array(y_te_pred_final)

# 预测误差：Pred - Actual
errors = y_pred - y_true

# 计算 RMSE 与 ±1 范围内的占比，方便在图上展示
rmse = np.sqrt(np.mean(errors ** 2))
within_1 = np.mean(np.abs(errors) <= 1.0)

print(f"RMSE (validation): {rmse:.3f}")
print(f"Proportion of |error| <= 1 star: {within_1:.3%}")

# 绘图
fig, ax = plt.subplots(figsize=(5.5, 4.0))

# 直方图（密度形式）
ax.hist(errors,
        bins=30,
        density=True,
        alpha=0.45,
        label='Error density')

# 垂直参考线：0 以及 ±1 星
ax.axvline(0, linestyle='-', linewidth=1.2, label='Error = 0')
ax.axvline(-1, linestyle='--', linewidth=1.1, label='Error = -1')
ax.axvline(1, linestyle='--', linewidth=1.1, label='Error = +1')

# 坐标与标题
ax.set_xlabel('Prediction error (Predicted - Actual)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('TEDM: Distribution of Prediction Errors (Validation, 5-fold CV)',
             fontsize=11, pad=8)

# 在图中写上 RMSE 和 “±1 星内比例”
text_x = errors.mean()
ax.text(text_x,
        ax.get_ylim()[1] * 0.9,
        f'RMSE = {rmse:.2f}\n'
        f'|error| ≤ 1 star: {within_1:.1%}',
        fontsize=9,
        ha='center',
        va='top',
        bbox=dict(boxstyle='round', alpha=0.08))

ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('fig_error_distribution_TEDM.png',
            dpi=300,
            bbox_inches='tight')
plt.show()

print("✅ Figure saved as 'fig_error_distribution_TEDM.png'")
