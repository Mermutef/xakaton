# tune_lgb.py – подбор гиперпараметров LightGBM через Optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('data/training_data.csv')
target = 'target'
features = [c for c in df.columns if c not in [target, 'ЛС', 'action']]
X = df[features]
y = df[target]
actions = df['action']

le = LabelEncoder()
action_encoded = le.fit_transform(actions)
X['action_encoded'] = action_encoded

tscv = TimeSeriesSplit(n_splits=5)


def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    rmse_list = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Лучшие параметры:", study.best_params)
print("Лучший RMSE:", study.best_value)

# Сохраняем лучшие параметры
joblib.dump(study.best_params, 'models/best_lgb_params.pkl')
