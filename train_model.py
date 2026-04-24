# train_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------
# 1. Загрузка данных
# ---------------------------
df = pd.read_csv("data/training_data.csv")
print(f"Датасет: {df.shape}")

target = "target"
features = [c for c in df.columns if c not in [target, "ЛС", "action"]]
X = df[features]
y = df[target]
actions = df["action"]

# Кодируем тип воздействия
le = LabelEncoder()
action_encoded = le.fit_transform(actions)
X["action_encoded"] = action_encoded
joblib.dump(le, "models/action_encoder.pkl")

# ---------------------------
# 2. Параметры LightGBM
# ---------------------------
# Автоматический баланс классов (если нужно)
neg, pos = y.value_counts().sort_index().values
scale = neg / pos  # у вас ~1.0, но пусть будет для страховки

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,  # немного уменьшили для регуляризации
    "max_depth": 12,  # ограничиваем глубину
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": scale,  # компенсация дисбаланса
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------
# 3. Кросс-валидация
# ---------------------------
tscv = TimeSeriesSplit(n_splits=5)
aucs, precisions, recalls, f1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)

    # Бинаризуем с порогом 0.5
    y_pred_binary = (y_pred > 0.5).astype(int)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)

    aucs.append(auc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(
        f"Fold {fold + 1} AUC: {auc:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}"
    )

print(f"\nСредний AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
print(f"Средний Precision: {np.mean(precisions):.4f}")
print(f"Средний Recall: {np.mean(recalls):.4f}")
print(f"Средний F1: {np.mean(f1s):.4f}")

# ---------------------------
# 4. Финальная модель на всех данных
# ---------------------------
full_train = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params,
    full_train,
    num_boost_round=int(2000 * 0.9),  # около среднего по фолдам
)
joblib.dump(final_model, "models/uplift_model.pkl")
joblib.dump(features, "models/feature_names.pkl")
print("Модель сохранена")

# ---------------------------
# 5. Важность признаков
# ---------------------------
importance = final_model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({"feature": X.columns, "importance": importance})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(30)
print("\nTop-30 важных признаков:")
print(feat_imp)

# ---------------------------
# 6. SHAP-анализ
# ---------------------------
import shap

explainer = shap.TreeExplainer(final_model)
sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
shap_values = explainer.shap_values(X.iloc[sample_idx])
joblib.dump(shap_values, "models/shap_values_sample.pkl")
joblib.dump(X.iloc[sample_idx], "models/shap_X_sample.pkl")
print("SHAP значения сохранены")
