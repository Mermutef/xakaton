import pandas as pd
import os

DATA_DIR = "data"


def read_general_info():
    """01 Общая информация о ЛС ХК_Лист1.csv"""
    filepath = os.path.join(DATA_DIR, "01 Общая информация о ЛС ХК_Лист1.csv")
    df = pd.read_csv(filepath, encoding='utf-8', sep=',', dtype=str)
    # Переименуем столбец ЛС в int
    df['ЛС'] = df['ЛС'].astype(int)
    # Все остальные столбцы - категориальные, оставим как str или преобразуем в bool
    return df


def read_turnover_balance():
    """02 Обортно-сальдовая ведомость ЛС ХК_Лист1.csv"""
    filepath = os.path.join(DATA_DIR, "02 Обортно-сальдовая ведомость ЛС ХК_Лист1.csv")
    # Читаем с мультииндексом: первая строка - даты, вторая - СЗ/Начислено/Опалачено
    df = pd.read_csv(filepath, encoding='utf-8', header=[0, 1])
    # Первый столбец - 'ЛС' c Unnamed второго уровня, переименуем
    # Объединяем названия столбцов: дата + суффикс
    new_cols = []
    for col in df.columns:
        if col[0] == 'ЛС':
            new_cols.append('ЛС')
        else:
            # убираем лишние пробелы, соединяем через пробел или _
            date_part = str(col[0]).strip()
            suffix_part = str(col[1]).strip()
            new_cols.append(f"{date_part}_{suffix_part}")
    df.columns = new_cols
    # Преобразуем 'ЛС' в int
    df['ЛС'] = pd.to_numeric(df['ЛС'], errors='coerce').fillna(0).astype(int)
    # Остальные столбцы - числа, запятые заменим на точки (если есть) и приведём к float
    for c in df.columns[1:]:
        # Проверим, является ли колонка строкой, и заменим запятые
        if df[c].dtype == object:
            df[c] = df[c].str.replace(',', '.').str.strip()
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df


def read_payments():
    """03 Оплаты ХК.csv"""
    filepath = os.path.join(DATA_DIR, "03 Оплаты ХК.csv")
    df = pd.read_csv(filepath, encoding='utf-8', sep=';', dtype=str)
    df.columns = ['ЛС', 'Дата оплаты', 'Сумма', 'Способ оплаты']
    df['ЛС'] = pd.to_numeric(df['ЛС'], errors='coerce').fillna(0).astype(int)
    df['Дата оплаты'] = pd.to_datetime(df['Дата оплаты'], dayfirst=True, errors='coerce')
    # Сумма: заменить запятую на точку, привести к float
    df['Сумма'] = df['Сумма'].str.replace(',', '.').astype(float)
    df['Способ оплаты'] = df['Способ оплаты'].astype(str).str.strip()
    return df


def read_measure(filename, measure_name_col):
    """Универсальное чтение файлов мер (04-13)"""
    filepath = os.path.join(DATA_DIR, filename)
    # Пропускаем первую строку, вторая строка - заголовки
    df = pd.read_csv(filepath, encoding='utf-8', skiprows=1, dtype=str)
    # Ожидаются столбцы: '{meas_name}','ЛС','Дата'   (на самом деле две колонки, но вторая - дата)
    # В данных видно: заголовок "Автодозвон," потом "ЛС,Дата". После skiprows=1 получим две колонки.
    # Названия колонок могут быть с пробелами, приведём к стандартным
    df.columns = ['ЛС', 'Дата']
    df['ЛС'] = pd.to_numeric(df['ЛС'], errors='coerce').dropna().astype(int)
    df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
    # удаляем строки с нулевым ЛС (если есть)
    df = df[df['ЛС'] != 0]
    return df


def load_all_data():
    """Загрузить все источники и вернуть словарь с DataFrame'ами"""
    data = {}
    data['general'] = read_general_info()
    data['turnover'] = read_turnover_balance()
    data['payments'] = read_payments()

    # Меры
    measures_files = {
        'autodial': '04 Автодозвон ХК_Лист1.csv',
        'email': '05 E-mail ХК_Лист1.csv',
        'sms': '06 СМС ХК_Лист1.csv',
        'operator_call': '07 Обзвон оператором ХК_Лист1.csv',
        'claim': '08 Претензия ХК_Лист1.csv',
        'visit': '09 Выезд к абоненту ХК_Лист1.csv',
        'notice_limit': '10 Уведомление о введении ограничения ХК_Лист1.csv',
        'limit': '11 Ограничение ХК_Лист1.csv',
        'court_order': '12 Заявление о выдаче судебного приказа ХК_Лист1.csv',
        'exec_doc': '13 Получение судебного приказа или ИЛ ХК_Лист1.csv'
    }
    for key, fname in measures_files.items():
        data[f'measure_{key}'] = read_measure(fname, key)

    # Лимиты
    limits_path = os.path.join(DATA_DIR, "14 Лимиты мер воздействия ХК_Лист1.csv")
    limits_df = pd.read_csv(limits_path, encoding='utf-8')
    limits_df.columns = ['Мера', 'Лимит']
    # уберем "нет ограничений" и преобразуем в int
    limits_df['Лимит'] = limits_df['Лимит'].replace('нет ограничений', '999999').astype(int)
    data['limits'] = limits_df

    return data


if __name__ == '__main__':
    # тестовый запуск
    all_data = load_all_data()
    for k, v in all_data.items():
        print(f"{k}: shape={v.shape}, columns={list(v.columns)[:5]}")
