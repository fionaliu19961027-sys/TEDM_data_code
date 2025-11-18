# -*- coding: utf-8 -*-
"""
TEDM 简化模型（无消费者个性特征: CP, CSI, UGP）
- 保留两阶段结构：基础属性非补偿门槛 + 其他属性补偿性加总
- 不考虑个体偏好权重矩阵、严格指数、一票否决个体差异、
  以及效用增益模式差异（UGP）
- 使用 Huber loss + CatBoost 残差修正
- 5 折交叉验证
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from catboost import CatBoostRegressor

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

df_reviews.columns = df_reviews.columns.astype(str).str.strip()
if "reviewer_id" in df_reviews.columns:
    df_reviews["reviewer_id"] = df_reviews["reviewer_id"].astype(str)

basic_attrs = ["Environment", "Price", "Waitstaff", "Service Attitude"]
other_attrs = ["Flavor", "Drinks", "Atmosphere", "Seating"]
all_attrs   = basic_attrs + other_attrs

control_vars = [
    "len_review_text",
    "num_punctuation",
    "num_senti_words",
    "total_attributes_found"
]

# 仅使用评论数据中的属性情感与控制变量
df = df_reviews.copy()
df["star_rate"] = pd.to_numeric(df["star_rate"], errors="coerce")

# 填补控制变量缺失
for c in control_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# 去除缺失的关键属性或评分
needed_cols = all_attrs + ["star_rate"]
df = df.dropna(subset=needed_cols)

def to_float(df_, cols):
    return df_[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

X_attr = to_float(df, all_attrs)
CTRL   = to_float(df, control_vars)
Y      = pd.to_numeric(df["star_rate"], errors="coerce").to_numpy(dtype=np.float32)

# -----------------------------
# 2) 控制变量标准化
# -----------------------------
scaler_ctrl = StandardScaler()
CTRL_scaled = scaler_ctrl.fit_transform(CTRL)

print("CTRL dim:", CTRL_scaled.shape[1])

# -----------------------------
# 3) K-Fold split (K=5)
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def tt(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=DEVICE)

# -----------------------------
# 4) Two-stage Simplified TEDM
# -----------------------------
class TwoStageModelSimple(nn.Module):
    """
    简化 TEDM:
    - 阶段1：基础属性一票否决 + 累积效用（无 CSI 异质性）
    - 阶段2：其他属性线性补偿（无 UGP 差异）
    - 控制变量线性修正
    """
    def __init__(self, basic_attr_count, other_attr_count, ctrl_dim):
        super().__init__()
        self.basic_attr_count = basic_attr_count
        self.other_attr_count = other_attr_count

        # 基础属性相关参数
        self.T_basic_raw = nn.Parameter(torch.zeros(basic_attr_count))  # ∈[-1,1]
        self.mu0_raw     = nn.Parameter(torch.tensor(0.0))             # ∈[1,3]
        self.kappa_raw   = nn.Parameter(torch.tensor(0.0))             # ∈[0,2]
        self.eta_raw     = nn.Parameter(torch.tensor(0.0))             # ∈[2.5,4]

        # 补偿阶段参数：统一线性效用
        self.beta_raw    = nn.Parameter(torch.zeros(other_attr_count)) # ≥0

        # 控制变量线性效应
        self.delta_ctrl  = nn.Linear(ctrl_dim, 1, bias=False)

        # Logistic 映射参数
        self.gamma0 = nn.Parameter(torch.tensor(0.0))
        self.gamma1 = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _sigmoid_bound(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _softplus_pos(raw, eps=1e-3):
        return torch.nn.functional.softplus(raw) + eps

    def forward(self, x, ctrl):
        # 参数有界化
        T_basic = self._sigmoid_bound(self.T_basic_raw, -1.0, 1.0)
        mu0     = self._sigmoid_bound(self.mu0_raw,   1.0, 3.0)
        kappa   = self._sigmoid_bound(self.kappa_raw, 0.0, 2.0)
        eta     = self._sigmoid_bound(self.eta_raw,   2.5, 4.0)
        beta    = self._softplus_pos(self.beta_raw)

        # ----- 阶段1：非补偿性 -----
        P_basic  = x[:, :self.basic_attr_count]
        T_expand = T_basic.unsqueeze(0).expand_as(P_basic)

        # 存在任一基础属性低于阈值 → 启动一票否决
        veto_mask = (P_basic < T_expand).any(dim=1)

        # 否决时：基于最大短缺深度惩罚
        penalty_depth = torch.relu(T_expand - P_basic).max(dim=1).values
        veto_score = mu0 - kappa * penalty_depth

        # 未否决时：基础属性的超阈值表现累积，并封顶于 eta
        base_effect = torch.relu(P_basic - T_expand).sum(dim=1)
        base_effect = torch.minimum(base_effect, eta.expand_as(base_effect))

        S_base = torch.where(veto_mask, veto_score, base_effect)

        # ----- 阶段2：补偿性 -----
        P_other = x[:, self.basic_attr_count:]
        # 统一线性补偿：不区分 UGP
        U_sum = torch.zeros_like(S_base)
        for i in range(self.other_attr_count):
            P_i = P_other[:, i]
            b_i = beta[i]
            U_sum = U_sum + b_i * P_i

        # 否决则仅保留 S_base，否则加入补偿效用
        S_total = torch.where(veto_mask, S_base, S_base + U_sum)

        # 控制变量线性修正
        ctrl_effect = self.delta_ctrl(ctrl).squeeze()
        S_final = S_total + ctrl_effect

        # Logistic 映射到 [1,5]
        y_pred = 1.0 + 4.0 / (1.0 + torch.exp(-(self.gamma0 + self.gamma1 * S_final)))
        return y_pred

# -----------------------------
# 5) K-Fold training & evaluation
# -----------------------------
EPOCHS = 200
LR = 1e-2
WD = 1e-4

fold_metrics = []

fold_id = 1
for train_idx, test_idx in kf.split(X_attr):
    print(f"\n================ Fold {fold_id}/5 ================\n")

    X_tr, X_te = X_attr[train_idx], X_attr[test_idx]
    CTRL_tr, CTRL_te = CTRL_scaled[train_idx], CTRL_scaled[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    X_tr_t    = tt(X_tr)
    X_te_t    = tt(X_te)
    CTRL_tr_t = tt(CTRL_tr)
    CTRL_te_t = tt(CTRL_te)
    Y_tr_t    = tt(Y_tr)
    Y_te_t    = tt(Y_te)

    torch.manual_seed(SEED)
    model = TwoStageModelSimple(len(basic_attrs), len(other_attrs), CTRL_tr.shape[1]).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    # 训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        y_hat = model(X_tr_t, CTRL_tr_t)
        loss = criterion(y_hat, Y_tr_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Fold {fold_id} | Epoch {epoch:03d}/{EPOCHS} | Train Loss={loss.item():.4f}")

    # -----------------------------
    # 6) CatBoost 残差修正（仅用属性+控制变量）
    # -----------------------------
    model.eval()
    with torch.no_grad():
        y_tr_pred_nn = model(X_tr_t, CTRL_tr_t).cpu().numpy()
        y_te_pred_nn = model(X_te_t, CTRL_te_t).cpu().numpy()

    residuals = Y_tr - y_tr_pred_nn
    # 残差模型输入：属性 + 控制变量（不含个性特征）
    X_train_features = np.hstack([X_tr, CTRL_tr])
    X_test_features  = np.hstack([X_te, CTRL_te])

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
    # 7) Evaluation
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

    tr_mse, tr_rmse, tr_mae, tr_r2 = report(f"训练集 (Simple TEDM + CatBoost) | Fold {fold_id}", Y_tr, y_tr_pred_final)
    te_mse, te_rmse, te_mae, te_r2 = report(f"验证集 (Simple TEDM + CatBoost) | Fold {fold_id}", Y_te, y_te_pred_final)

    fold_metrics.append((tr_mse, tr_rmse, tr_mae, tr_r2,
                         te_mse, te_rmse, te_mae, te_r2))

    fold_id += 1

# -----------------------------
# 8) 汇总 5 折均值与标准差
# -----------------------------
metrics_arr = np.array(fold_metrics)
means = metrics_arr.mean(axis=0)
stds  = metrics_arr.std(axis=0)

print("\n================ 5-Fold Summary (Simple TEDM) ================\n")
names = ["Train MSE", "Train RMSE", "Train MAE", "Train R2",
         "Val MSE", "Val RMSE", "Val MAE", "Val R2"]
for i, name in enumerate(names):
    print(f"{name}: {means[i]:.6f} ± {stds[i]:.6f}")
