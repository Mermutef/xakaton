# build_training_dataset.py (финальная версия без ошибок)
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from data_loader import load_all_data
from feature_engineering import build_static_features

REFERENCE_DATE = pd.Timestamp('2026-03-01')


# ------------------------------------------------------------
# 1. Макроэкономические показатели
# ------------------------------------------------------------
def get_key_rate_by_month():
    """Возвращает словарь {pd.Timestamp('YYYY-MM-01'): ставка}"""
    df = pd.read_csv(
        'data/IR_CHG_MPOFull_таб_дни изменений_полная версия.csv',
        encoding='utf-8-sig', sep=',', dtype=str, skiprows=2
    )
    df.columns = ['Дата', 'Ключевая ставка'] + [f'col{i}' for i in range(2, len(df.columns))]
    df = df[['Дата', 'Ключевая ставка']].dropna()
    df['Дата'] = pd.to_datetime(df['Дата'].str.strip(), errors='coerce')
    df['Ключевая ставка'] = pd.to_numeric(df['Ключевая ставка'].str.replace(',', '.'), errors='coerce')
    df = df[df['Дата'] >= pd.Timestamp('2020-01-01')].sort_values('Дата')

    months = pd.date_range('2020-01-01', '2026-05-01', freq='MS')
    rates = {}
    for m in months:
        prev = df[df['Дата'] <= m]
        if not prev.empty:
            rates[m] = prev.iloc[-1]['Ключевая ставка']
        else:
            rates[m] = None
    return rates


def get_inflation_by_month():
    """Возвращает словарь {pd.Timestamp('YYYY-MM-01'): инфляция % г/г}"""
    infl = pd.read_csv(
        'data/Инфляция и ключевая ставка Банка России_F25_10_2025_T23_04_2026_Инфляция и ключевая ставка Банк.csv',
        encoding='utf-8-sig', sep=',', dtype=str
    )
    infl.columns = infl.columns.str.replace('"', '').str.strip()
    dates = infl.iloc[:, 0]
    values = infl.iloc[:, 2]
    inflation_dict = {}
    for d, v in zip(dates, values):
        if pd.isna(d):
            continue
        parts = str(d).strip().split('.')
        if len(parts) == 2:
            month, year = int(parts[0]), int(parts[1])
            if 1 <= month <= 12:
                date_key = pd.Timestamp(year=year, month=month, day=1)
                inflation_dict[date_key] = float(str(v).replace(',', '.'))
    all_months = pd.date_range('2020-01-01', '2026-05-01', freq='MS')
    full_infl = {}
    for m in all_months:
        full_infl[m] = inflation_dict.get(m, np.nan)
    infl_series = pd.Series(full_infl)
    infl_series = infl_series.fillna(infl_series.mean())
    return infl_series.to_dict()


# ------------------------------------------------------------
# 2. Вычисление динамических признаков на конкретный месяц
# ------------------------------------------------------------
def compute_turnover_features_at_month(turnover_df, target_month):
    """Признаки из оборотки на начало target_month (данные строго до него)."""
    import re
    date_pat = re.compile(r'^(\d{4}-\d{2}-\d{2})')
    month_types = {}
    for col in turnover_df.columns[1:]:
        match = date_pat.match(col)
        if match:
            date_str = match.group(1)
            if '.' in col[len(date_str):len(date_str) + 15]:
                date_str = '2026-03-01'
            if date_str not in month_types:
                month_types[date_str] = {}
            suffix = col.split('_', 1)[1] if '_' in col else col[len(date_str):].strip()
            month_types[date_str][suffix] = col

    months = sorted(month_types.keys())
    past_months = [m for m in months if m < target_month.strftime('%Y-%m-%d')]
    if not past_months:
        return pd.DataFrame({
            'ЛС': turnover_df['ЛС'], 'debt_current': 0.0, 'months_debt': 0,
            'payment_ratio_6m': 0.0, 'max_debt_12m': 0.0,
            'avg_accrual_6m': 0.0, 'trend_slope': 0.0, 'debt_std': 0.0
        })

    last_month = past_months[-1]
    if 'СЗ на начало' in month_types[last_month]:
        debt_current = turnover_df[month_types[last_month]['СЗ на начало']]
    else:
        debt_current = 0.0

    def consecutive_debt(row):
        cnt = 0
        for m in reversed(past_months):
            if 'СЗ на начало' in month_types[m]:
                if row[month_types[m]['СЗ на начало']] > 0:
                    cnt += 1
                else:
                    break
        return cnt

    months_debt = turnover_df.apply(consecutive_debt, axis=1)

    recent_months = [m for m in past_months if m >= (target_month - pd.DateOffset(months=6)).strftime('%Y-%m-%d')]

    def payment_ratio_row(row):
        pays = []
        for m in recent_months:
            if 'Опалачено' in month_types[m]:
                pays.append(row[month_types[m]['Опалачено']] > 0)
        return np.mean(pays) if pays else 0.0

    payment_ratio_6m = turnover_df.apply(payment_ratio_row, axis=1)

    debt_cols_12 = [month_types[m]['СЗ на начало'] for m in past_months
                    if m >= (target_month - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
                    and 'СЗ на начало' in month_types[m]]
    if debt_cols_12:
        max_debt_12m = turnover_df[debt_cols_12].max(axis=1)
    else:
        max_debt_12m = debt_current

    nach_cols = [month_types[m]['Начислено'] for m in recent_months if 'Начислено' in month_types[m]]
    avg_accrual_6m = turnover_df[nach_cols].mean(axis=1) if nach_cols else 0.0

    sz_cols = [month_types[m]['СЗ на начало'] for m in past_months if 'СЗ на начало' in month_types[m]]
    if len(sz_cols) >= 2:
        x = np.arange(len(sz_cols))

        def slope(row):
            y = row[sz_cols].values.astype(float)
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return m

        trend_slope = turnover_df.apply(slope, axis=1)
    else:
        trend_slope = 0.0

    debt_std = turnover_df[sz_cols].std(axis=1) if sz_cols else 0.0

    features = pd.DataFrame({
        'ЛС': turnover_df['ЛС'],
        'debt_current': debt_current,
        'months_debt': months_debt,
        'payment_ratio_6m': payment_ratio_6m,
        'max_debt_12m': max_debt_12m,
        'avg_accrual_6m': avg_accrual_6m,
        'trend_slope': trend_slope,
        'debt_std': debt_std
    })
    return features


def compute_payment_features_at_month(payments_df, target_month):
    """Признаки из оплат до начала target_month."""
    recent = payments_df[payments_df['Дата оплаты'] < target_month]
    if recent.empty:
        return pd.DataFrame({'ЛС': payments_df['ЛС'].unique(), 'total_paid_6m': 0.0,
                             'median_payment': 0.0, 'count_payments_6m': 0,
                             'mode_5_ratio': 0.0, 'days_since_last_payment': 9999})
    agg = recent.groupby('ЛС').agg(
        total_paid_6m=('Сумма', 'sum'),
        median_payment=('Сумма', 'median'),
        count_payments_6m=('Сумма', 'count'),
        mode_5_ratio=('Способ оплаты', lambda x: (x == '5').mean())
    ).reset_index()
    last = recent.groupby('ЛС')['Дата оплаты'].max().reset_index()
    last['days_since_last_payment'] = (target_month - last['Дата оплаты']).dt.days
    result = pd.merge(agg, last, on='ЛС', how='right').fillna(0)
    return result


def compute_measure_history_at_month(measure_dfs, target_month):
    """Признаки по истории мер строго до начала target_month."""
    feats = pd.DataFrame({'ЛС': pd.concat([df['ЛС'] for df in measure_dfs.values()]).unique()})
    for name, mdf in measure_dfs.items():
        hist = mdf[mdf['Дата'] < target_month]
        total = hist.groupby('ЛС').size().rename(f'cnt_{name}')
        recent = hist[hist['Дата'] >= target_month - pd.DateOffset(months=3)].groupby('ЛС').size().rename(
            f'cnt_recent_{name}')
        last = hist.groupby('ЛС')['Дата'].max().rename(f'last_date_{name}')
        feats = feats.merge(total, on='ЛС', how='left')
        feats = feats.merge(recent, on='ЛС', how='left')
        feats = feats.merge(last, on='ЛС', how='left')
        feats[f'cnt_{name}'] = feats[f'cnt_{name}'].fillna(0).astype(int)
        feats[f'cnt_recent_{name}'] = feats[f'cnt_recent_{name}'].fillna(0).astype(int)
        feats[f'days_since_last_{name}'] = (target_month - feats[f'last_date_{name}']).dt.days.fillna(9999).astype(int)
        feats = feats.drop(columns=[f'last_date_{name}'])
    info_measures = ['autodial', 'email', 'sms', 'operator_call', 'claim', 'visit', 'notice_limit']
    feats['stage'] = 0
    has_info = feats[[f'cnt_{m}' for m in info_measures]].sum(axis=1) > 0
    feats.loc[has_info, 'stage'] = 1
    has_limit = feats['cnt_limit'] > 0
    feats.loc[has_limit, 'stage'] = 2
    has_court = (feats['cnt_court_order'] + feats['cnt_exec_doc']) > 0
    feats.loc[has_court, 'stage'] = 3
    return feats


def compute_features_at_month(target_month, all_data, static_feats, clusters_df, key_rate_dict, infl_dict):
    turnover_feat = compute_turnover_features_at_month(all_data['turnover'], target_month)
    payment_feat = compute_payment_features_at_month(all_data['payments'], target_month)
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
    measure_feat = compute_measure_history_at_month(measure_dfs, target_month)
    static = static_feats.copy()
    cluster_col = clusters_df[['ЛС', 'cluster']].dropna()

    master = static.merge(turnover_feat, on='ЛС', how='left')
    master = master.merge(payment_feat, on='ЛС', how='left')
    master = master.merge(measure_feat, on='ЛС', how='left')
    master = master.merge(cluster_col, on='ЛС', how='left')
    master = master.fillna(0)
    master['key_rate'] = key_rate_dict.get(target_month, np.nan)
    master['inflation'] = infl_dict.get(target_month, np.nan)
    master = master.loc[:, ~master.columns.duplicated()]
    num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = ['ЛС'] + [c for c in num_cols if c != 'ЛС']
    master = master[numeric_cols]
    master = master.drop_duplicates(subset='ЛС', keep='first').reset_index(drop=True)
    return master


# ------------------------------------------------------------
# 3. Формирование обучающей выборки с таргетом
# ------------------------------------------------------------
def build_training_data():
    print("Загрузка сырых данных...")
    all_data = load_all_data()
    static_feats = build_static_features(all_data['general'])
    clusters_df = pd.read_csv('data/clusters.csv')
    key_rates = get_key_rate_by_month()
    infl_rates = get_inflation_by_month()

    months = pd.date_range('2025-02-01', '2026-01-01', freq='MS')
    all_samples = []

    for cur_month in months:
        print(f"Обработка месяца {cur_month.strftime('%Y-%m')}")
        feats = compute_features_at_month(cur_month, all_data, static_feats, clusters_df, key_rates, infl_rates)
        next_month = cur_month + pd.DateOffset(months=1)

        # Меры в текущем месяце (как и раньше)
        measures_in_month = {}
        for mname, mdf in {
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
        }.items():
            mask = (mdf['Дата'] >= cur_month) & (mdf['Дата'] < next_month)
            measures_in_month[mname] = mdf.loc[mask, 'ЛС'].unique()

        all_ls = set()
        for arr in measures_in_month.values():
            all_ls.update(arr)
        if not all_ls:
            continue

        # ---------- НОВОЕ: определение успеха по факту оплаты ----------
        payments_df = all_data['payments']
        # Оплаты в следующем календарном месяце
        payments_next = payments_df[
            (payments_df['Дата оплаты'] >= next_month) &
            (payments_df['Дата оплаты'] < next_month + pd.DateOffset(months=1))
            ]
        paid_ls = set(payments_next['ЛС'].unique())

        # Для ускорения строим словари
        feats_dict = {}
        for _, r in feats.iterrows():
            feats_dict[r['ЛС']] = r.to_dict()

        for ls in all_ls:
            row = feats_dict.get(ls)
            if row is None:
                continue
            # Цель: 1 – была оплата в следующем месяце, иначе 0
            target = 1 if ls in paid_ls else 0
            for mname, ls_list in measures_in_month.items():
                if ls in ls_list:
                    sample = row.copy()
                    sample['action'] = mname
                    sample['target'] = target
                    all_samples.append(sample)
        # ----------------------------------------------------------------

    train_df = pd.DataFrame(all_samples)
    # Проверим баланс
    print("Распределение целевой переменной:")
    print(train_df['target'].value_counts())
    train_df.to_csv('data/training_data.csv', index=False)
    print(f"Обучающая выборка сохранена: {train_df.shape}")
    return train_df


if __name__ == '__main__':
    build_training_data()
