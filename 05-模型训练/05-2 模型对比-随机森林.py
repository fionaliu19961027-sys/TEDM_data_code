import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1) 读取数据
df_reviews = pd.read_excel("model_training_data.xlsx")
df_prefs   = pd.read_excel("consumer_preference_matrix.xlsx")
df_strict  = pd.read_excel("strictness_analysis.xlsx")
df_utility = pd.read_excel("enhanced_utility_analysis.xlsx")

# 基础属性和补偿属性
basic_attrs = ["Environment", "Price", "Waitstaff", "Service Attitude"]
other_attrs = ["Flavor", "Drinks", "Atmosphere", "Seating"]
all_attrs   = basic_attrs + other_attrs

# 偏好矩阵重命名，避免重名
pref_cols = [c for c in df_prefs.columns if c != "reviewer_id"]
df_prefs = df_prefs.rename(columns={c: f"{c}_w" for c in pref_cols})

# 合并
df = (df_reviews
      .merge(df_prefs, on="reviewer_id", how="inner")
      .merge(df_strict, on="reviewer_id", how="inner")
      .merge(df_utility, on="reviewer_id", how="inner"))

df = df.dropna()

# 2) 绘制评分分布直方图
plt.figure(figsize=(6,4))
df["star_rate"].hist(bins=np.arange(0.5, 6.5, 1), rwidth=0.8)
plt.xlabel("Star Rate")
plt.ylabel("Count")
plt.title("Distribution of Star Ratings")
plt.show()

# 3) 构建基线模型输入
X = df[all_attrs].values
y = df["star_rate"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) 训练随机森林回归
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5) 预测与评估
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\n=== 随机森林基线模型 ===")
print("\n训练集：")
print("MSE :", mean_squared_error(y_train, y_pred_train))
print("RMSE:", rmse(y_train, y_pred_train))
print("MAE :", mean_absolute_error(y_train, y_pred_train))
print("R2  :", r2_score(y_train, y_pred_train))

print("\n测试集：")
print("MSE :", mean_squared_error(y_test, y_pred_test))
print("RMSE:", rmse(y_test, y_pred_test))
print("MAE :", mean_absolute_error(y_test, y_pred_test))
print("R2  :", r2_score(y_test, y_pred_test))
