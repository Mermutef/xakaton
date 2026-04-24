# train_model.py – регрессия на снижение долга + метрики + сравнение с CatBoost
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ---------- Загрузка данных ----------
df = pd.read_csv('data/training_data.csv')
print(f"Датасет: {df.shape}")

target = 'target'
features = [c for c in df.columns if c not in [target, 'ЛС', 'action']]
X = df[features]
y = df[target]
actions = df['action']

le = LabelEncoder()
action_encoded = le.fit_transform(actions)
X['action_encoded'] = action_encoded
joblib.dump(le, 'models/action_encoder.pkl')

# ---------- Параметры LightGBM ----------
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'max_depth': 12,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# ---------- Кросс-валидация LightGBM ----------
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
        num_boost_round=2000,
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

# ---------- Анализ ошибок LightGBM на всей выборке ----------
print("\n=== Анализ ошибок LightGBM (на всей выборке) ===")
full_train = lgb.Dataset(X, label=y)
final_lgb = lgb.train(lgb_params, full_train, num_boost_round=int(2000 * 0.9))
y_pred_lgb = final_lgb.predict(X)
errors_lgb = y_pred_lgb - y
print(f"Средняя ошибка (bias): {errors_lgb.mean():.2f}")
print(f"Медианная ошибка: {np.median(errors_lgb):.2f}")
print(f"Стандартное отклонение ошибки: {errors_lgb.std():.2f}")
print(f"5-й процентиль ошибки: {np.percentile(errors_lgb, 5):.2f}")
print(f"95-й процентиль ошибки: {np.percentile(errors_lgb, 95):.2f}")

# Сохраняем LightGBM как основную модель
joblib.dump(final_lgb, 'models/uplift_model.pkl')
joblib.dump(features, 'models/feature_names.pkl')
print("Модель LightGBM сохранена (models/uplift_model.pkl)")

# Важность признаков LightGBM
importance = final_lgb.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(30)
print("\nTop-30 важных признаков LightGBM:")
print(feat_imp)

# SHAP для LightGBM
import shap

explainer = shap.TreeExplainer(final_lgb)
sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
shap_values = explainer.shap_values(X.iloc[sample_idx])
joblib.dump(shap_values, 'models/shap_values_sample.pkl')
joblib.dump(X.iloc[sample_idx], 'models/shap_X_sample.pkl')
print("SHAP значения (LightGBM) сохранены")

# ---------- Сравнение с CatBoost ----------
print("\n" + "=" * 60)
print("Сравнение с CatBoostRegressor")
print("=" * 60)
try:
    from catboost import CatBoostRegressor

    cat_features = ['cluster', 'stage']  # дополнительно можно 'action_encoded', если хотим

    # Кросс-валидация CatBoost
    cb_rmse_scores, cb_mae_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cb_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features  # передаём индексы или названия категориальных колонок
        )
        cb_model.fit(X_train, y_train)
        y_pred_cb = cb_model.predict(X_val)
        rmse_cb = np.sqrt(mean_squared_error(y_val, y_pred_cb))
        mae_cb = mean_absolute_error(y_val, y_pred_cb)
        cb_rmse_scores.append(rmse_cb)
        cb_mae_scores.append(mae_cb)

    print(f"CatBoost средний RMSE: {np.mean(cb_rmse_scores):.2f} (±{np.std(cb_rmse_scores):.2f})")
    print(f"CatBoost средний MAE : {np.mean(cb_mae_scores):.2f} (±{np.std(cb_mae_scores):.2f})")

    # Сравнение
    if np.mean(cb_rmse_scores) < np.mean(rmse_scores):
        print("\nCatBoost показал лучший RMSE, можно переключиться на него.")
        # Обучим на всех данных и сохраним
        final_cb = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=100,
            allow_writing_files=False,
            cat_features=cat_features
        )
        final_cb.fit(X, y)
        # Если нужно заменить основную модель, раскомментируй:
        # joblib.dump(final_cb, 'models/uplift_model.pkl')
        # print("Основная модель заменена на CatBoost (models/uplift_model.pkl)")
    else:
        print("\nLightGBM остаётся лучшей моделью.")

except ImportError:
    print("CatBoost не установлен. Пропускаем сравнение.")
