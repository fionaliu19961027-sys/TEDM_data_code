# -*- coding: utf-8 -*-
"""
Two-stage consumer rating model with residual correction
- NN with Huber loss
- Residual correction with CatBoost
- Stage-2 gating: if veto, skip compensation (S_total = S_base)
- K=5 交叉验证版本
- 本版本增加：strict_thresh (τ) 参数敏感性分析：
    τ ∈ {None(学习), 0.4, 0.5, 0.6}
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
df_prefs   = pd.read_excel(DATA_DIR / "consumer_preference_matrix.xlsx")
df_strict  = pd.read_excel(DATA_DIR / "strictness_analysis.xlsx")
df_utility = pd.read_excel(DATA_DIR / "enhanced_utility_analysis.xlsx")

# 清洗列名 & reviewer_id 类型
for d in (df_reviews, df_prefs, df_strict, df_utility):
    d.columns = d.columns.astype(str).str.strip()
    if "reviewer_id" in d.columns:
        d["reviewer_id"] = d["reviewer_id"].astype(str)

basic_attrs = ["Environment", "Price", "Waitstaff", "Service Attitude"]
other_attrs = ["Flavor", "Drinks", "Atmosphere", "Seating"]
all_attrs   = basic_attrs + other_attrs
control_vars = ["len_review_text", "num_punctuation", "num_senti_words", "total_attributes_found"]

# 偏好矩阵列重命名为 *_w
pref_cols = [c for c in df_prefs.columns if c != "reviewer_id"]
df_prefs = df_prefs.rename(columns={c: f"{c}_w" for c in pref_cols})

# 合并：以 reviewer_id 为键左连接
df = (df_reviews
      .merge(df_prefs, on="reviewer_id", how="left")
      .merge(df_strict, on="reviewer_id", how="left")
      .merge(df_utility, on="reviewer_id", how="left"))

df["star_rate"] = pd.to_numeric(df["star_rate"], errors="coerce")

utility_map = {"乐观型": 0, "理性型": 1, "挑剔型": 2}
df["utility_type"] = df["final_utility_type"].map(utility_map)

# 控制变量缺失填 0
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
# 2) Standardize + PCA for PREFS & CTRL
# -----------------------------
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
# 3) K-Fold split
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def tt(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=DEVICE)


# -----------------------------
# 4) Two-stage NN Model
#    加入 fixed_tau，用于敏感性分析
# -----------------------------
class TwoStageModel(nn.Module):
    def __init__(self, basic_attr_count, other_attr_count, pref_dim, ctrl_dim, fixed_tau=None):
        super().__init__()
        self.basic_attr_count = basic_attr_count
        self.other_attr_count = other_attr_count
        self.fixed_tau = fixed_tau  # 若不为 None，则使用固定 τ

        # trainable & bounded via transforms
        self.strict_thresh_raw = nn.Parameter(torch.tensor(0.0))            # → [0,1]（仅在 fixed_tau=None 时生效）
        self.T_basic_raw       = nn.Parameter(torch.zeros(basic_attr_count))# → [-1,1]
        self.mu0_raw   = nn.Parameter(torch.tensor(0.0))   # → [1,3]
        self.kappa_raw = nn.Parameter(torch.tensor(0.0))   # → [0,2]
        self.eta_raw   = nn.Parameter(torch.tensor(0.0))   # → [2.5,4]

        self.beta_raw  = nn.Parameter(torch.zeros(other_attr_count))  # ≥0
        self.alpha_raw = nn.Parameter(torch.zeros(other_attr_count))  # >0

        self.delta_pref = nn.Linear(pref_dim, 1, bias=False)
        self.delta_ctrl = nn.Linear(ctrl_dim, 1, bias=False)

        self.gamma0 = nn.Parameter(torch.tensor(0.0))
        self.gamma1 = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _sigmoid_bound(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _softplus_pos(raw, eps=1e-3):
        return torch.nn.functional.softplus(raw) + eps

    def forward(self, x, prefs, strictness, utility_type, ctrl):
        # ---- 严格阈值 τ ----
        if self.fixed_tau is not None:
            strict_thresh = torch.tensor(
                float(self.fixed_tau),
                dtype=torch.float32,
                device=x.device
            )
        else:
            strict_thresh = self._sigmoid_bound(self.strict_thresh_raw, 0.0, 1.0)

        # ---- 其他参数做有界变换 ----
        T_basic = self._sigmoid_bound(self.T_basic_raw, -1.0, 1.0)
        mu0     = self._sigmoid_bound(self.mu0_raw,   1.0, 3.0)
        kappa   = self._sigmoid_bound(self.kappa_raw, 0.0, 2.0)
        eta     = self._sigmoid_bound(self.eta_raw,   2.5, 4.0)
        beta    = self._softplus_pos(self.beta_raw)
        alpha   = self._softplus_pos(self.alpha_raw)
        eps     = 1e-6

        # ----- Stage 1: Non-compensatory -----
        P_basic  = x[:, :self.basic_attr_count]
        T_expand = T_basic.unsqueeze(0).expand_as(P_basic)

        # 是否有基础属性未达标
        has_shortfall = (P_basic < T_expand).any(dim=1)
        # 最大踩雷深度
        penalty_depth = torch.relu(T_expand - P_basic).max(dim=1).values
        # 严格到足以否决？
        veto_strict = (strictness > strict_thresh)
        # 一票否决条件
        veto_mask = has_shortfall & veto_strict

        # 否决时低效用
        veto_score = mu0 - kappa * penalty_depth

        # 未否决时累积基础效用并封顶
        base_effect = torch.relu(P_basic - T_expand).sum(dim=1)
        base_effect = torch.minimum(base_effect, eta.expand_as(base_effect))

        S_base = torch.where(veto_mask, veto_score, base_effect)

        # ----- Stage 2: Compensatory (仅未否决样本参与) -----
        P_other = x[:, self.basic_attr_count:]
        U_sum = torch.zeros_like(S_base)

        for i in range(self.other_attr_count):
            P_i = P_other[:, i]
            b_i = beta[i]
            a_i = alpha[i]

            U_lin   = b_i * P_i
            U_opt   = b_i * (torch.exp(a_i * P_i) - 1.0) / (torch.exp(a_i) - 1.0 + eps)
            U_picky = b_i * (1.0 - torch.exp(-a_i * P_i)) / (1.0 - torch.exp(-a_i) + eps)

            U_sel = torch.where(
                utility_type == 0, U_opt,
                torch.where(utility_type == 1, U_lin, U_picky)
            )
            U_sum = U_sum + U_sel

        # 否决样本跳过补偿：S_total = S_base；其他样本：S_base + U_sum
        S_total = torch.where(veto_mask, S_base, S_base + U_sum)

        # ----- 控制变量线性效应 -----
        ctrl_effect = self.delta_pref(prefs).squeeze() + self.delta_ctrl(ctrl).squeeze()
        S_final = S_total + ctrl_effect

        # ----- Logistic mapping to [1,5] -----
        y_pred = 1.0 + 4.0 / (1.0 + torch.exp(-(self.gamma0 + self.gamma1 * S_final)))
        return y_pred


# -----------------------------
# 5) Tau sensitivity: K-Fold training & evaluation
# -----------------------------
EPOCHS = 200
LR = 1e-2
WD = 1e-4

# None = 原始模型（τ 可学习），其余为固定阈值
tau_list = [None, 0.4, 0.5, 0.6]
all_results = []

for tau in tau_list:
    label = "Learned tau (主模型)" if tau is None else f"Fixed tau = {tau:.2f}"
    print("\n" + "=" * 40)
    print(f" Tau sensitivity run: {label}")
    print("=" * 40)

    fold_metrics = []
    fold_id = 1

    for train_idx, test_idx in kf.split(X_attr):
        print(f"\n---- Fold {fold_id}/5 ----")

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

        # 初始化模型（根据 tau 是否固定）
        torch.manual_seed(SEED)
        model = TwoStageModel(
            basic_attr_count=len(basic_attrs),
            other_attr_count=len(other_attrs),
            pref_dim=PREFS_tr.shape[1],
            ctrl_dim=CTRL_tr.shape[1],
            fixed_tau=tau
        ).to(DEVICE)

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        # 训练
        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            y_hat = model(X_tr_t, PREFS_tr_t, STR_tr_t, UTYPE_tr_t, CTRL_tr_t)
            loss = criterion(y_hat, Y_tr_t)
            loss.backward()
            optimizer.step()

        # NN 预测
        model.eval()
        with torch.no_grad():
            y_tr_pred_nn = model(X_tr_t, PREFS_tr_t, STR_tr_t, UTYPE_tr_t, CTRL_tr_t).cpu().numpy()
            y_te_pred_nn = model(X_te_t, PREFS_te_t, STR_te_t, UTYPE_te_t, CTRL_te_t).cpu().numpy()

        # CatBoost 拟合残差
        residuals = Y_tr - y_tr_pred_nn
        X_train_features = np.hstack([X_tr, PREFS_tr, CTRL_tr,
                                      STR_tr.reshape(-1, 1),
                                      UTYPE_tr.reshape(-1, 1)])
        X_test_features  = np.hstack([X_te, PREFS_te, CTRL_te,
                                      STR_te.reshape(-1, 1),
                                      UTYPE_te.reshape(-1, 1)])

        cat = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.02,
            l2_leaf_reg=3,
            random_seed=SEED,
            loss_function="RMSE",
            verbose=False
        )
        cat.fit(X_train_features, residuals)
        residuals_pred_train = cat.predict(X_train_features)
        residuals_pred_test  = cat.predict(X_test_features)

        y_tr_pred_final = y_tr_pred_nn + residuals_pred_train
        y_te_pred_final = y_te_pred_nn + residuals_pred_test

        # 指标
        mse_tr = mean_squared_error(Y_tr, y_tr_pred_final)
        rmse_tr = np.sqrt(mse_tr)
        mae_tr = mean_absolute_error(Y_tr, y_tr_pred_final)
        r2_tr = r2_score(Y_tr, y_tr_pred_final)

        mse_te = mean_squared_error(Y_te, y_te_pred_final)
        rmse_te = np.sqrt(mse_te)
        mae_te = mean_absolute_error(Y_te, y_te_pred_final)
        r2_te = r2_score(Y_te, y_te_pred_final)

        fold_metrics.append((mse_tr, rmse_tr, mae_tr, r2_tr,
                             mse_te, rmse_te, mae_te, r2_te))

        print(f"Fold {fold_id} | Val RMSE={rmse_te:.4f} | Val R2={r2_te:.4f}")
        fold_id += 1

    # 汇总 5 折均值 & 标准差
    metrics_arr = np.array(fold_metrics)
    means = metrics_arr.mean(axis=0)
    stds  = metrics_arr.std(axis=0)

    # 汇总 5 折均值 & 标准差
    metrics_arr = np.array(fold_metrics)
    means = metrics_arr.mean(axis=0)
    stds = metrics_arr.std(axis=0)

    all_results.append({
        "label": label,
        # Train
        "train_rmse_mean": means[1],
        "train_rmse_std": stds[1],
        "train_r2_mean": means[3],
        "train_r2_std": stds[3],
        # Test / Validation
        "test_rmse_mean": means[5],
        "test_rmse_std": stds[5],
        "test_r2_mean": means[7],
        "test_r2_std": stds[7],
    })

    print(f"\nSummary ({label})")
    names = ["Train MSE", "Train RMSE", "Train MAE", "Train R2",
             "Val MSE", "Val RMSE", "Val MAE", "Val R2"]
    for i, name in enumerate(names):
        print(f"{name}: {means[i]:.6f} ± {stds[i]:.6f}")

# -----------------------------
# 6) Tau 敏感性总表
# -----------------------------
print("\n========== Tau Sensitivity Summary (Train & Test, 5-fold means) ==========")
print("Model\t\t\tTrain RMSE\tTrain R²\tTest RMSE\tTest R²")
for r in all_results:
    print(
        f"{r['label']}"
        f"\t{r['train_rmse_mean']:.3f}±{r['train_rmse_std']:.3f}"
        f"\t{r['train_r2_mean']:.3f}±{r['train_r2_std']:.3f}"
        f"\t{r['test_rmse_mean']:.3f}±{r['test_rmse_std']:.3f}"
        f"\t{r['test_r2_mean']:.3f}±{r['test_r2_std']:.3f}"
    )
