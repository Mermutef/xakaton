# explain.py – персональное объяснение рекомендации (человеческим языком)
import pandas as pd
import numpy as np
import joblib
import sys
import os
from data_loader import load_all_data
from feature_engineering import build_static_features
from build_training_dataset import (
    compute_turnover_features_at_month,
    compute_payment_features_at_month,
    compute_measure_history_at_month
)
from config import ACTION_NAMES, FEATURE_HUMAN
import warnings

warnings.filterwarnings('ignore')

# Месяц возьмём из файла рекомендаций (первая строка `target_month`? его нет, но можно по файлу)
# Проще: из переменной окружения TARGET_MONTH, иначе дефолт
target_str = os.environ.get("TARGET_MONTH", "2026-03-01")
TARGET_MONTH = pd.Timestamp(target_str)


def compute_features_for_ls(ls_number):
    """Вычисляет признаки на текущий целевой месяц для заданного ЛС."""
    all_data = load_all_data()
    static_feats = build_static_features(all_data['general'])
    clusters = pd.read_csv('data/clusters.csv')
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
    master = master.loc[:, ~master.columns.duplicated()]
    num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = ['ЛС'] + [c for c in num_cols if c != 'ЛС']
    master = master[numeric_cols]
    master = master.drop_duplicates(subset='ЛС', keep='first').reset_index(drop=True)
    if 'cluster' in master.columns:
        master['cluster'] = master['cluster'].astype(int)
    if 'stage' in master.columns:
        master['stage'] = master['stage'].astype(int)
    row = master[master['ЛС'] == ls_number]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def explain_recommendation(ls_number):
    # Пытаемся найти в рекомендациях
    recs = pd.read_csv('data/recommendations.csv')
    row = recs[recs['ЛС'] == ls_number]
    if row.empty:
        print(f"Для ЛС {ls_number} рекомендация не найдена.")
        return
    row = row.iloc[0]

    action_name = ACTION_NAMES.get(row['action'], row['action'])
    print(f"ЛС: {ls_number}")
    print(f"Кластер: {row['cluster']}")
    print(f"Назначенная мера: {action_name}")
    print(f"Ожидаемый возврат (снижение долга): {row['expected_return']:,.2f} руб.")
    print(f"Текущий долг: {row['debt_current']:,.2f} руб.")
    print(f"Месяцев задолженности: {row['months_debt']}")

    feat_dict = compute_features_for_ls(ls_number)
    if feat_dict is None:
        print("Не удалось вычислить признаки для этого ЛС.")
        return

    # Портрет должника
    phone = feat_dict.get('Наличие телефона', 0)
    email_kvit = feat_dict.get('электронная квитанция', 0)
    city = feat_dict.get('Город', 0)
    chd = feat_dict.get('ЧД', 0)
    mkd = feat_dict.get('МКД', 0)
    addr_freq = feat_dict.get('address_freq', 0)
    print("\nХарактеристики должника:")
    print(f"  Телефон: {'есть' if phone == 1 else 'нет'}")
    print(f"  Электронная квитанция (e-mail): {'есть' if email_kvit == 1 else 'нет'}")
    print(f"  Тип жилья: {'частный дом' if chd == 1 else 'многоквартирный дом' if mkd == 1 else 'неизвестно'}")
    print(f"  Проживание: {'город' if city == 1 else 'сельская местность'}")
    print(f"  Количество ЛС по этому адресу: {addr_freq:.0f}")

    model = joblib.load('models/uplift_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    if 'action_encoded' not in feature_names:
        feature_names = list(feature_names) + ['action_encoded']

    action_encoder = joblib.load('models/action_encoder.pkl')
    action_to_idx = {a: i for i, a in enumerate(action_encoder.classes_)}
    if row['action'] not in action_to_idx:
        print(f"Действие {row['action']} не найдено в энкодере.")
        return
    feat_dict['action_encoded'] = action_to_idx[row['action']]

    X = pd.DataFrame([feat_dict])[feature_names]
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    pred = model.predict(X)[0]
    shap_contribs = model.predict(X, pred_contrib=True)[0]
    base_value = shap_contribs[-1]
    shap_values = shap_contribs[:-1]

    feat_df = pd.DataFrame({
        'feature': feature_names,
        'value': X.values[0],
        'shap': shap_values
    })
    feat_df['abs_shap'] = feat_df['shap'].abs()
    top5 = feat_df.nlargest(5, 'abs_shap')

    print(f"\nБазовое (среднее) снижение долга: {base_value:,.2f} руб.")
    print("Топ-5 факторов, объясняющих предсказание:\n")
    for _, f in top5.iterrows():
        human_name, comment = FEATURE_HUMAN.get(f['feature'], (f['feature'], ''))
        direction = "повышает" if f['shap'] >= 0 else "понижает"
        print(f"  • {human_name}: {f['value']:,.2f}")
        print(f"    {direction} ожидаемый возврат на {abs(f['shap']):,.2f} руб.")
        if comment:
            print(f"    ({comment})")
        print()

    # Итоговая интерпретация
    print("=" * 60)
    print("Интерпретация:")
    if row['expected_return'] > base_value:
        print("Модель считает, что этот должник вернёт БОЛЬШЕ среднего, потому что:")
    else:
        print("Модель считает, что этот должник вернёт МЕНЬШЕ среднего, потому что:")
    for _, f in top5.iterrows():
        human_name, _ = FEATURE_HUMAN.get(f['feature'], (f['feature'], ''))
        if f['shap'] >= 0:
            print(f"  + {human_name} способствует возврату")
        else:
            print(f"  – {human_name} уменьшает ожидаемый возврат")
    print(f"\nИтоговый прогноз: {row['expected_return']:,.2f} руб. снижения долга "
          f"при назначении меры '{action_name}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ls = input("Введите номер ЛС: ")
    else:
        ls = sys.argv[1]
    explain_recommendation(int(ls))
