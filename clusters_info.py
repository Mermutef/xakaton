# clusters_info.py – описание профилей кластеров с автоматической интерпретацией
import pandas as pd
import numpy as np
from config import ACTION_NAMES, FEATURE_HUMAN


def interpret_cluster(row, count):
    """Генерирует текстовое описание кластера на основе средних значений признаков."""
    debt = max(0, row['debt_current'])
    months = max(0, row['months_debt'])
    pay_ratio = max(0, min(1, row['payment_ratio_6m']))
    accr = max(0, row['avg_accrual_6m'])
    stage = max(0, min(3, round(row['stage'])))
    addr_freq = max(0, row['address_freq'])
    cnt_autodial = max(0, row['cnt_autodial'])
    cnt_notice = max(0, row['cnt_notice_limit'])
    cnt_claim = max(0, row.get('cnt_claim', 0))

    # Оценка уровня долга
    if debt < 500:
        debt_level = "низкий"
    elif debt < 3000:
        debt_level = "средний"
    elif debt < 10000:
        debt_level = "высокий"
    else:
        debt_level = "очень высокий"

    # Оценка срока задолженности
    if months < 3:
        maturity = "свежая"
    elif months < 7:
        maturity = "умеренная"
    elif months < 12:
        maturity = "застарелая"
    else:
        maturity = "хроническая"

    # Платёжная дисциплина
    if pay_ratio > 0.7:
        discipline = "хорошая (регулярные платежи)"
    elif pay_ratio > 0.4:
        discipline = "средняя (платит эпизодически)"
    else:
        discipline = "плохая (почти нет оплат)"

    # Стадия взыскания
    stage_desc = {0: "без применённых мер", 1: "информирование", 2: "ограничение",
                  3: "судебное/исполнительное производство"}
    stage_text = stage_desc.get(stage, "неизвестно")

    # Интенсивность предыдущих мер
    total_measures = cnt_autodial + cnt_notice + cnt_claim
    if total_measures < 1:
        workload = "практически не применялись"
    elif total_measures < 3:
        workload = "применялись умеренно"
    else:
        workload = "применялись активно"

    # Дополнительные статические признаки (усреднённые по кластеру)
    phone = max(0, min(1, row.get('Наличие телефона', 0)))
    email_kvit = max(0, min(1, row.get('электронная квитанция', 0)))
    city = max(0, min(1, row.get('Город', 0)))
    chd = max(0, min(1, row.get('ЧД', 0)))
    mkd = max(0, min(1, row.get('МКД', 0)))

    description = (
        f"Кластер объединяет {count} должников.\n"
        f"Средний долг {debt:,.0f} руб. ({debt_level}) при среднемесячном начислении {accr:,.0f} руб.\n"
        f"Задолженность {maturity} (в среднем {months:.1f} мес.), платёжная дисциплина {discipline}.\n"
        f"Стадия взыскания: {stage_text}. Меры воздействия {workload}.\n"
        f"Частота адреса (количество лицевых счетов на адрес): {addr_freq:.0f}.\n"
        f"Наличие телефона: {phone:.0%}, e-mail квитанция: {email_kvit:.0%}.\n"
        f"Город: {city:.0%}, частный дом: {chd:.0%}, многоквартирный дом: {mkd:.0%}.\n"
    )

    # Практические рекомендации
    if pay_ratio > 0.5 and debt < 3000:
        description += "Этот сегмент хорошо реагирует на лёгкие напоминания (E-mail, СМС).\n"
    elif pay_ratio < 0.3 and debt > 5000:
        description += "Рекомендуются более жёсткие меры: уведомления, ограничения, судебное взыскание.\n"
    elif stage >= 2:
        description += "Должники находятся на поздних стадиях, необходимы исполнительные действия.\n"
    else:
        description += "Подойдут персональные обзвоны и автодозвон.\n"

    return description


def describe_clusters():
    centers = pd.read_csv('data/cluster_centers.csv')
    clusters = pd.read_csv('data/clusters.csv')
    cluster_counts = clusters['cluster'].value_counts().sort_index()

    print("=== ПРОФИЛИ КЛАСТЕРОВ ДОЛЖНИКОВ ===\n")
    for idx, row in centers.iterrows():
        cl = idx
        count = cluster_counts.get(cl, 0)
        print(f"--- Кластер {cl} ({count} чел.) ---")
        print(interpret_cluster(row, count))
        print()


if __name__ == "__main__":
    describe_clusters()
