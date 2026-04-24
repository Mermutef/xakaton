import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

df = pd.read_csv('data/training_data.csv')
print(f"Датасет до обработки выбросов: {df.shape}")

lower = df['target'].quantile(0.01)
upper = df['target'].quantile(0.99)
df = df[(df['target'] >= lower) & (df['target'] <= upper)]
print(f"Датасет после удаления выбросов: {df.shape}")

target = 'target'
features = [c for c in df.columns if c not in [target, 'ЛС', 'action']]
X = df[features]
y = df[target]
actions = df['action']

le = LabelEncoder()
action_encoded = le.fit_transform(actions)
X['action_encoded'] = action_encoded
X['cluster'] = X['cluster'].astype(int)
X['stage'] = X['stage'].astype(int)
joblib.dump(le, 'models/action_encoder.pkl')

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 126,
    'max_depth': 11,
    'learning_rate': 0.0832,
    'min_child_samples': 39,
    'reg_alpha': 0.5362,
    'reg_lambda': 0.0841,
    'feature_fraction': 0.8522,
    'bagging_fraction': 0.9701,
    'bagging_freq': 10,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores, mae_scores, r2_scores, ev_scores = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    ev = explained_variance_score(y_val, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    ev_scores.append(ev)

    print(f"Fold {fold + 1} RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, EV: {ev:.4f}")

print("\n=== LightGBM средние метрики ===")
print(f"RMSE : {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
print(f"MAE  : {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")
print(f"R²   : {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
print(f"EV   : {np.mean(ev_scores):.4f} (±{np.std(ev_scores):.4f})")

X_train_full, X_valid, y_train_full, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=False  # сохраняем временной порядок
)
train_data = lgb.Dataset(X_train_full, label=y_train_full)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

print("\nОбучение финальной модели с early stopping...")
final_lgb = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=3000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)
print(f"Оптимальное число итераций: {final_lgb.best_iteration}")

print("\n=== Анализ ошибок финальной модели ===")
y_pred_full = final_lgb.predict(X)
errors = y_pred_full - y
print(f"Средняя ошибка (bias): {errors.mean():.2f}")
print(f"Медианная ошибка: {np.median(errors):.2f}")
print(f"Стандартное отклонение ошибки: {errors.std():.2f}")
print(f"5-й процентиль ошибки: {np.percentile(errors, 5):.2f}")
print(f"95-й процентиль ошибки: {np.percentile(errors, 95):.2f}")

joblib.dump(final_lgb, 'models/uplift_model.pkl')
joblib.dump(features, 'models/feature_names.pkl')
print("Модель LightGBM сохранена (models/uplift_model.pkl)")

importance = final_lgb.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(30)
print("\nTop-30 важных признаков:")
print(feat_imp)

import shap

explainer = shap.TreeExplainer(final_lgb)
sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
shap_values = explainer.shap_values(X.iloc[sample_idx])
joblib.dump(shap_values, 'models/shap_values_sample.pkl')
joblib.dump(X.iloc[sample_idx], 'models/shap_X_sample.pkl')
print("SHAP значения сохранены")

import json

metrics = {
    'R2': round(np.mean(r2_scores), 4),
    'RMSE': round(np.mean(rmse_scores), 2),
    'MAE': round(np.mean(mae_scores), 2),
}
with open('models/training_metrics.json', 'w') as f:
    json.dump(metrics, f)
print("Метрики сохранены в models/training_metrics.json")
