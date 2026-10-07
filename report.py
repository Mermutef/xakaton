import pandas as pd


def generate_report():
    print("=" * 60)
    print("СЕГМЕНТАЦИЯ ДОЛЖНИКОВ")
    print("=" * 60)
    try:
        clusters = pd.read_csv("data/clusters.csv")
        print(f"Количество кластеров: {clusters['cluster'].nunique()}")
        for cl, grp in clusters.groupby('cluster'):
            print(f"  Кластер {cl}: {len(grp)} должников")
        print()
    except Exception as e:
        print(f"Ошибка загрузки кластеров: {e}")

    print("=" * 60)
    print("РЕКОМЕНДАЦИИ НА МАРТ 2026")
    print("=" * 60)
    try:
        recs = pd.read_csv("data/recommendations_march2026.csv")
        print(f"Всего назначено мер: {len(recs)}")
        print(f"Средний ожидаемый возврат: {recs['expected_return'].mean():.2f} руб.")
        print(f"Суммарный ожидаемый возврат: {recs['expected_return'].sum():.2f} руб.")
        print()
        print("Распределение мер:")
        print(recs['action'].value_counts().to_string())
        print()
        print("Средние характеристики по мерам:")
        print(recs.groupby('action').agg(
            avg_debt=('debt_current', 'mean'),
            avg_months=('months_debt', 'mean'),
            count=('ЛС', 'count')
        ).round(2))
    except Exception as e:
        print(f"Ошибка загрузки рекомендаций: {e}")


if __name__ == "__main__":
    generate_report()
