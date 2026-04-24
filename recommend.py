# recommend.py – финальная версия с жадным алгоритмом
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

from data_loader import load_all_data
from feature_engineering import build_static_features
from build_training_dataset import (
    compute_turnover_features_at_month,
    compute_payment_features_at_month,
    compute_measure_history_at_month
)

TARGET_MONTH = pd.Timestamp('2026-03-01')
KEY_RATE_MAR2026 = 15.0
INFLATION_MAR2026 = 5.86

# 1. Загрузка данных и модели
print("Загрузка данных и модели...")
all_data = load_all_data()
static_feats = build_static_features(all_data['general'])
clusters = pd.read_csv('data/clusters.csv')

model = joblib.load('models/uplift_model.pkl')
action_encoder = joblib.load('models/action_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')
if 'action_encoded' not in feature_names:
    feature_names = list(feature_names) + ['action_encoded']


# 2. Признаки на март 2026
def compute_features_current():
    print("Вычисление признаков на март 2026...")
    turnover_feat = compute_turnover_features_at_month(all_data['turnover'], TARGET_MONTH)
    payment_feat = compute_payment_features_at_month(all_data['payments'], TARGET_MONTH)
    measure_dfs = {
        'autodial': all_data['measure_autodial'],
        'email': all_data['measure_email'],
        'sms': all_data['measure_sms'],
        'operator_call': all_data['measure_operator_call'],
        'claim': all_data['measure_claim'],
        'visit': all_data['measure_visit'],
        'notice_limit': all_data['measure_notice_limit'],
        'limit': all_data['measure_limit'],
        'court_order': all_data['measure_court_order'],
        'exec_doc': all_data['measure_exec_doc']
    }
    measure_feat = compute_measure_history_at_month(measure_dfs, TARGET_MONTH)
    static = static_feats.copy()
    cluster_col = clusters[['ЛС', 'cluster']].dropna()

    master = static.merge(turnover_feat, on='ЛС', how='left')
    master = master.merge(payment_feat, on='ЛС', how='left')
    master = master.merge(measure_feat, on='ЛС', how='left')
    master = master.merge(cluster_col, on='ЛС', how='left')
    master = master.fillna(0)
    master['key_rate'] = KEY_RATE_MAR2026
    master['inflation'] = INFLATION_MAR2026
    master = master.loc[:, ~master.columns.duplicated()]
    num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = ['ЛС'] + [c for c in num_cols if c != 'ЛС']
    master = master[numeric_cols]
    master = master.drop_duplicates(subset='ЛС', keep='first').reset_index(drop=True)
    return master


current_features = compute_features_current()


# 3. Допустимые меры
def determine_allowed_actions(df):
    allowed = []
    for _, row in df.iterrows():
        debt = row['debt_current']
        age_debt = row['months_debt']
        has_phone = row['Наличие телефона'] == 1
        has_email = row['электронная квитанция'] == 1
        stage = row['stage']
        acts = []
        if debt >= 500 and has_email:
            acts.append('email')
        if 500 <= debt <= 1500 and 2 <= age_debt <= 6 and has_phone:
            acts.append('autodial')
        if 500 <= debt <= 2000 and 1 <= age_debt <= 2 and has_phone:
            acts.append('sms')
        if debt >= 1500 and 2 <= age_debt <= 6 and has_phone:
            acts.append('operator_call')
        if debt >= 500 and 2 <= age_debt <= 6 and not has_phone:
            acts.append('claim')
        if debt >= 1500 and age_debt >= 1 and stage >= 1:
            acts.append('notice_limit')
        if ((age_debt >= 11 and debt >= 2000) or
            (age_debt >= 6 and debt >= 4000) or
            (age_debt >= 4 and debt >= 10000)) and stage >= 2:
            acts.append('court_order')
        if debt > 2000 and stage >= 3:
            acts.append('exec_doc')
        allowed.append(acts)
    return allowed


current_features['allowed_actions'] = determine_allowed_actions(current_features)

# 4. Предсказание вероятностей (безопасный сбор)
action_classes = list(action_encoder.classes_)
action_to_idx = {a: i for i, a in enumerate(action_classes)}

print("Предсказание вероятностей...")
records = current_features.to_dict(orient='records')
ls_list, action_list, prob_list, debt_list = [], [], [], []

for rec in records:
    ls_val = rec['ЛС']
    actions = rec['allowed_actions']
    if not actions:
        continue
    feat_dict = {k: v for k, v in rec.items() if k not in ['target', 'allowed_actions']}
    for act in actions:
        if act not in action_to_idx:
            continue
        code = action_to_idx[act]
        feat_dict['action_encoded'] = code
        X = pd.DataFrame([feat_dict])[feature_names]
        prob = model.predict(X)[0]
        ls_list.append(ls_val)
        action_list.append(act)
        prob_list.append(prob)
        debt_list.append(rec['debt_current'])

pred_df = pd.DataFrame({
    'ЛС': ls_list,
    'action': action_list,
    'prob_success': prob_list,
    'debt_current': debt_list
})

print(f"Сгенерировано {len(pred_df)} рекомендаций")
print(f"Распределение prob_success: min={pred_df['prob_success'].min():.4f}, "
      f"mean={pred_df['prob_success'].mean():.4f}, max={pred_df['prob_success'].max():.4f}")
print(f"Количество с prob > 0.5: {(pred_df['prob_success'] > 0.5).sum()}")

# Отсеиваем совсем бесперспективные (опционально, порог можно менять)
prob_threshold = 0.001
pred_df = pred_df[pred_df['prob_success'] > prob_threshold].copy()
print(f"После фильтра prob > {prob_threshold}: {len(pred_df)} записей")

# 5. Жадное назначение с учётом лимитов
LIMITS = {
    'email': 999999,
    'sms': 2150,
    'autodial': 8000,
    'operator_call': 1550,
    'claim': 400,
    'visit': 500,
    'notice_limit': 6200,
    'limit': 200,
    'court_order': 400,
    'exec_doc': 250
}

# Вычисляем ожидаемую выгоду и сортируем для каждой меры
pred_df['expected_return'] = pred_df['prob_success'] * pred_df['debt_current']

# Порядок применения мер: сначала массовые с большим лимитом, потом дефицитные
measure_order = ['email', 'autodial', 'notice_limit', 'sms', 'operator_call', 'claim', 'visit', 'court_order', 'limit',
                 'exec_doc']

assigned = set()  # ЛС, которые уже получили какую-то меру
assignments = []

for measure in measure_order:
    limit = LIMITS.get(measure, 0)
    if limit == 0:
        continue
    candidates = pred_df[pred_df['action'] == measure].copy()
    # исключаем уже назначенных
    candidates = candidates[~candidates['ЛС'].isin(assigned)]
    # сортируем по ожидаемой выгоде
    candidates = candidates.sort_values('expected_return', ascending=False)
    # выбираем не более limit записей
    selected = candidates.head(limit)
    for _, row in selected.iterrows():
        assignments.append({'ЛС': row['ЛС'], 'action': measure,
                            'prob_success': row['prob_success'],
                            'expected_return': row['expected_return']})
        assigned.add(row['ЛС'])

result_df = pd.DataFrame(assignments)
print(f"Назначено мер: {len(result_df)}")
if not result_df.empty:
    print(result_df['action'].value_counts())

# 6. Обогащение и сохранение
result_df = result_df.merge(
    current_features[['ЛС', 'cluster', 'debt_current', 'months_debt']],
    on='ЛС', how='left'
)
# expected_return уже есть в result_df из цикла, но если нет – пересчитаем
if 'expected_return' not in result_df.columns:
    result_df = result_df.merge(pred_df[['ЛС', 'action', 'expected_return']], on=['ЛС', 'action'], how='left')

result_df.to_csv('data/recommendations_march2026.csv', index=False)
print("Рекомендации сохранены в data/recommendations_march2026.csv")

print("\nОтчёт по использованию лимитов:")
for m in measure_order:
    limit = LIMITS.get(m, 0)
    used = result_df[result_df['action'] == m].shape[0]
    print(f"{m}: использовано {used}/{limit}")

print("\nСредние характеристики назначенных должников:")
print(result_df.groupby('action').agg(
    avg_debt=('debt_current', 'mean'),
    avg_months=('months_debt', 'mean'),
    count=('ЛС', 'count')
))
