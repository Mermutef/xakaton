# run.py
import argparse
from pathlib import Path
import subprocess
import sys
import pandas as pd
import os


def convert_xlsx_to_csv(data_dir: Path):
    """Запускает скрипт конвертации xlsx -> csv, если есть xlsx файлы."""
    xlsx_files = list(data_dir.glob("*.xlsx"))
    if not xlsx_files:
        return
    print("Найдены xlsx файлы, конвертация...")
    # Используем отдельный скрипт конвертации (можно и встроить)
    subprocess.run([sys.executable, "convert_xlsx.py", str(data_dir)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Рекомендательная система для хакатона")
    parser.add_argument("--mode", choices=["train", "infer", "explain", "report", "clusters"], required=True,
                        help="Режим: train – обучение модели, infer – получение рекомендаций")
    parser.add_argument("--data", default="data", help="Папка с исходными данными")
    parser.add_argument("--models", default="models", help="Папка для сохранения/загрузки моделей")
    parser.add_argument("--target", default="2026-03-01", help="Целевой месяц для рекомендаций (YYYY-MM-01)")
    args = parser.parse_args()

    data_path = Path(args.data)
    models_path = Path(args.models)
    target_month = pd.Timestamp(args.target)

    # Создаем папки при необходимости
    models_path.mkdir(exist_ok=True)

    # Конвертация xlsx -> csv (если есть)
    convert_xlsx_to_csv(data_path)

    # Проверяем наличие необходимых CSV
    required_files = [
        "01 Общая информация о ЛС ХК_Лист1.csv",
        "02 Обортно-сальдовая ведомость ЛС ХК_Лист1.csv",
        "03 Оплаты ХК.csv",
        "14 Лимиты мер воздействия ХК_Лист1.csv",
    ]
    for f in required_files:
        if not (data_path / f).exists():
            raise FileNotFoundError(f"Не найден файл {f} в папке {data_path}")

    # Добавим путь в sys.path для импорта модулей проекта
    sys.path.insert(0, str(Path(__file__).parent))

    if args.mode == "train":
        print("=== Обучение модели ===")
        from build_training_dataset import build_training_data
        # Обучающая выборка
        build_training_data()
        # Обучаем модель
        subprocess.run([sys.executable, "train_model.py"], check=True)
        print("Обучение завершено, модель сохранена в models/uplift_model.pkl")

    elif args.mode == "infer":
        print(f"=== Получение рекомендаций на {target_month.strftime('%Y-%m')} ===")
        # Проверяем наличие обученной модели
        model_file = models_path / "uplift_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError("Модель не найдена. Сначала запустите обучение: --mode train")

        # Импортируем и запускаем recommend.py с нужным месяцем
        # Мы можем просто выполнить recommend.py, передав target_month через временную переменную окружения
        os.environ["TARGET_MONTH"] = args.target
        subprocess.run([sys.executable, "recommend.py"], check=True)
        print("Рекомендации сохранены в data/recommendations_march2026.csv")
    elif args.mode == "explain":
        ls = input("Введите номер ЛС: ")
        subprocess.run([sys.executable, "explain.py", ls])
    elif args.mode == "report":
        subprocess.run([sys.executable, "report.py"])
    elif args.mode == "clusters":
        subprocess.run([sys.executable, "clusters_info.py"])


if __name__ == "__main__":
    main()
