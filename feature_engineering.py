import pandas as pd
import numpy as np
from data_loader import load_all_data

REFERENCE_DATE = pd.Timestamp(
    '2026-03-01')  # на этот месяц будем давать рекомендации, используем данные до конца февраля 2026
LAST_MONTH = '2026-02-01'  # последний закрытый месяц в оборотке


def build_static_features(general_df):
    """Преобразование статических признаков из 01 таблицы"""
    df = general_df.copy()
    # Бинарные признаки
    binary_cols = [
        'Возможность дистанционного отключения',
        'Наличие телефона',
        'Наличие льгот',
        'Газификация дома',
        'Город',
        'ЯрОблИЕРЦ квитанция',
        'Почта России квитанция',
        'электронная квитанция',
        'не проживает',
        'ЧД',
        'МКД',
        'Общежитие',
        'Установка Тамбур',
        'Установка опора',
        'Установка в квартире/доме',
        'Установка лестничкая клетка'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Да': 1, 'Нет': 0})
    # Адрес: заменим на частоту (сколько ЛС с таким же ГУИД)
    freq = df['Адрес (ГУИД)'].value_counts()
    df['address_freq'] = df['Адрес (ГУИД)'].map(freq).fillna(1).astype(int)
    # Удалим исходный ГУИД
    df = df.drop(columns=['Адрес (ГУИД)'])
    return df


def extract_turnover_features(turnover_df):
    """Извлечение признаков из оборотки: текущий долг, тренды, история оплат"""
    df = turnover_df.copy()
    # Разбор колонок: извлекаем дату (первые 10 символов) и суффикс (всё после '_')
    import re
    date_pat = re.compile(r'^(\d{4}-\d{2}-\d{2})')
    month_types = {}
    for col in df.columns[1:]:
        # Ищем дату в начале строки (10 символов YYYY-MM-DD)
        match = date_pat.match(col)
        if match:
            date_str = match.group(1)
            # Суффикс - всё остальное после первого пробела и '_'? У нас имена типа "2025-01-01 00:00:00_СЗ на начало"
            # Проще: отделим суффикс как часть после первого '_', если он есть
            if '_' in col:
                suffix = col.split('_', 1)[1]
            else:
                suffix = col[len(date_str):].strip()  # fallback
            # Для столбцов с точкой в дате (например "2026-02-01 00:00:00.3") date_str всё равно "2026-02-01" по первым 10 символам,
            # но тогда мы потеряем уникальность. Определим, есть ли точка: если после пробела идёт точка, это дубль.
            if '.' in col[len(date_str):len(date_str) + 15]:  # примерно
                # Это дополнительный месяц, считаем его мартом
                date_str = '2026-03-01'
            if date_str not in month_types:
                month_types[date_str] = {}
            month_types[date_str][suffix] = col
        else:
            # возможно, колонка без даты (например, "Unnamed: 0"), пропускаем
            pass

    months = sorted(month_types.keys())
    # Создаём признаки
    features = pd.DataFrame({'ЛС': df['ЛС']})
    # Текущий долг (конец февраля 2026): используем СЗ на начало марта, если есть
    if '2026-03-01' in month_types and 'СЗ на начало' in month_types['2026-03-01']:
        features['debt_current'] = df[month_types['2026-03-01']['СЗ на начало']]
    else:
        # Расчёт через февраль
        if '2026-02-01' in month_types:
            feb = month_types['2026-02-01']
            features['debt_current'] = df[feb['СЗ на начало']] + df[feb['Начислено']] - df[feb['Опалачено']]
        else:
            features['debt_current'] = 0.0
    # Срок непрерывной задолженности
    debt_cols = [col for date in months if date <= '2026-02-01' for col in [month_types[date].get('СЗ на начало')] if
                 col]

    # Вычисляем наличие долга (СЗ > 0) последовательно с конца
    def consecutive_debt(row):
        cnt = 0
        for date in reversed(months):
            if date > '2026-02-01':
                continue
            if 'СЗ на начало' in month_types[date]:
                if row[month_types[date]['СЗ на начало']] > 0:
                    cnt += 1
                else:
                    break
        return cnt

    features['months_debt'] = df.apply(consecutive_debt, axis=1)
    # Доля месяцев с оплатой > 0 за последние 6 мес (2025-09..2026-02)
    recent_months = [m for m in months if '2025-09' <= m <= '2026-02']

    def payment_ratio_row(row):
        pays = []
        for m in recent_months:
            if 'Опалачено' in month_types[m]:
                pays.append(row[month_types[m]['Опалачено']] > 0)
        return np.mean(pays) if pays else 0.0

    features['payment_ratio_6m'] = df.apply(payment_ratio_row, axis=1)
    # Максимальный долг за последние 12 мес
    debt_end_cols = [month_types[m]['СЗ на начало'] for m in months if
                     m >= '2025-03-01' and m <= '2026-02-01' and 'СЗ на начало' in month_types[m]]
    if debt_end_cols:
        features['max_debt_12m'] = df[debt_end_cols].max(axis=1)
    else:
        features['max_debt_12m'] = features['debt_current']
    # Среднее начисление за последние 6 мес
    nach_cols = [month_types[m]['Начислено'] for m in recent_months if 'Начислено' in month_types[m]]
    if nach_cols:
        features['avg_accrual_6m'] = df[nach_cols].mean(axis=1)
    else:
        features['avg_accrual_6m'] = 0.0
    # Тренд долга (наклон по СЗ на начало)
    sz_cols = [month_types[m]['СЗ на начало'] for m in months if 'СЗ на начало' in month_types[m] and m <= '2026-02-01']
    x = np.arange(len(sz_cols))

    def slope(row):
        y = row[sz_cols].values
        if len(y) < 2:
            return 0.0
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    features['trend_slope'] = df[sz_cols].apply(slope, axis=1)
    # Волатильность долга
    features['debt_std'] = df[sz_cols].std(axis=1)
    return features


def extract_payment_features(payments_df):
    """Признаки из истории оплат"""
    # Агрегируем по ЛС
    # Суммируем за последние 6 мес, медианный платеж, кол-во платежей, способ оплаты 5
    cutoff = REFERENCE_DATE - pd.DateOffset(months=6)
    recent = payments_df[payments_df['Дата оплаты'] >= cutoff]
    agg = recent.groupby('ЛС').agg(
        total_paid_6m=('Сумма', 'sum'),
        median_payment=('Сумма', 'median'),
        count_payments_6m=('Сумма', 'count'),
        mode_5_ratio=('Способ оплаты', lambda x: (x == '5').mean())
    ).reset_index()
    # Добавим давность последней оплаты
    last_payment = payments_df.groupby('ЛС')['Дата оплаты'].max().reset_index()
    last_payment['days_since_last_payment'] = (REFERENCE_DATE - last_payment['Дата оплаты']).dt.days
    # Мёрджим
    result = pd.merge(agg, last_payment, on='ЛС', how='right')
    result = result.fillna(0)
    return result


def extract_measure_history(measure_dfs):
    """Создание признаков по истории мер: количество, давность, стадия"""
    # measure_dfs - словарь {название: df}
    all_measures = {}
    for name, mdf in measure_dfs.items():
        mdf = mdf.copy()
        mdf = mdf.rename(columns={'Дата': f'date_{name}'})
        all_measures[name] = mdf
    # Собираем все ЛС
    ls_set = set()
    for df in all_measures.values():
        ls_set.update(df['ЛС'].unique())
    features = pd.DataFrame({'ЛС': list(ls_set)})
    # Для каждой меры: кол-во применений, кол-во за последние 3 мес, давность последнего
    for name, mdf in all_measures.items():
        total = mdf.groupby('ЛС').size().rename(f'cnt_{name}')
        recent = mdf[mdf[f'date_{name}'] >= REFERENCE_DATE - pd.DateOffset(months=3)].groupby('ЛС').size().rename(
            f'cnt_recent_{name}')
        last = mdf.groupby('ЛС')[f'date_{name}'].max().rename(f'last_date_{name}')
        features = features.merge(total, on='ЛС', how='left')
        features = features.merge(recent, on='ЛС', how='left')
        features = features.merge(last, on='ЛС', how='left')
        features[f'cnt_{name}'] = features[f'cnt_{name}'].fillna(0).astype(int)
        features[f'cnt_recent_{name}'] = features[f'cnt_recent_{name}'].fillna(0).astype(int)
        # Давность последнего в днях
        features[f'days_since_last_{name}'] = (REFERENCE_DATE - features[f'last_date_{name}']).dt.days
        features[f'days_since_last_{name}'] = features[f'days_since_last_{name}'].fillna(9999).astype(int)
        features = features.drop(columns=[f'last_date_{name}'])
    # Определение стадии (максимальной процедуры)
    # Информирование: autodial, email, sms, operator_call, claim, visit, notice_limit
    # Ограничение: limit
    # Взыскание судебное: court_order, exec_doc
    features['stage'] = 0  # нет мер
    # Если хоть одно уведомление или информирование было (кроме limit, court_order, exec_doc) => стадия 1
    info_measures = ['autodial', 'email', 'sms', 'operator_call', 'claim', 'visit', 'notice_limit']
    has_info = features[[f'cnt_{m}' for m in info_measures]].sum(axis=1) > 0
    features.loc[has_info, 'stage'] = 1
    # Если было ограничение (limit) -> стадия 2
    has_limit = features['cnt_limit'] > 0
    features.loc[has_limit, 'stage'] = 2
    # Если было судебное (court_order или exec_doc) -> стадия 3
    has_court = (features['cnt_court_order'] + features['cnt_exec_doc']) > 0
    features.loc[has_court, 'stage'] = 3
    return features


def build_master_table():
    """Объединяет все признаки в один DataFrame на каждый ЛС"""
    all_data = load_all_data()
    static = build_static_features(all_data['general'])
    turnover_feat = extract_turnover_features(all_data['turnover'])
    payment_feat = extract_payment_features(all_data['payments'])
    measure_feat = extract_measure_history({
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
    })
    # Объединяем все по ЛС
    master = static.merge(turnover_feat, on='ЛС', how='left')
    master = master.merge(payment_feat, on='ЛС', how='left')
    master = master.merge(measure_feat, on='ЛС', how='left')
    # Заполняем пропуски
    master = master.fillna(0)
    # Оставляем только числовые столбцы (все строковые должны были быть преобразованы)
    numeric_cols = ['ЛС'] + master.select_dtypes(include=[np.number]).columns.tolist()
    master = master[numeric_cols]
    return master


if __name__ == '__main__':
    master_df = build_master_table()
    print(f"Master table shape: {master_df.shape}")
    print(master_df.head())
    # Сохраним для дальнейшего использования
    master_df.to_csv('data/master_features.csv', index=False)
