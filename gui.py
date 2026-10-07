# gui.py – финальная версия с выбором месяца и человеческими названиями
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import subprocess
import sys
import os
import json
import joblib

st.set_page_config(page_title="Рекомендательная система", layout="wide")

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Подключаем человеческие названия мер
from config import ACTION_NAMES


# ------------------ Конвертация xlsx ------------------
def convert_all_xlsx(data_dir):
    xlsx_files = list(Path(data_dir).glob("*.xlsx"))
    if not xlsx_files:
        return
    subprocess.run([sys.executable, "convert_xlsx.py", str(data_dir)], check=True)


# ------------------ Определение доступных месяцев ------------------
@st.cache_data
def get_available_months():
    file = DATA_DIR / "02 Обортно-сальдовая ведомость ЛС ХК_Лист1.csv"
    if not file.exists():
        return []
    df = pd.read_csv(file, encoding='utf-8', header=[0, 1], nrows=0)
    months_set = set()
    for col in df.columns:
        if col[0] == 'ЛС':
            continue
        date_str = str(col[0]).strip()
        try:
            dt = pd.to_datetime(date_str)
            months_set.add(dt.to_period('M').to_timestamp())
        except:
            pass
    months = sorted(months_set)
    if months:
        last_data_month = months[-1]
        target_months = pd.date_range(
            start=months[0] + pd.DateOffset(months=1),
            end=last_data_month + pd.DateOffset(months=1),
            freq='MS'
        )
        return list(target_months)  # <-- вот это изменение
    return []


# ------------------ Загрузка рекомендаций ------------------
@st.cache_data
def load_recommendations():
    file = DATA_DIR / "recommendations.csv"
    if not file.exists():
        return None
    df = pd.read_csv(file)
    df['action_name'] = df['action'].map(ACTION_NAMES)
    return df


# ------------------ Боковая панель: загрузка файлов ------------------
st.sidebar.title("Загрузка исходных данных")
uploaded_files = st.sidebar.file_uploader(
    "Выберите файлы (Excel или CSV)",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Сохранить загруженные файлы"):
        with st.sidebar.spinner("Сохраняем..."):
            for uf in uploaded_files:
                file_path = DATA_DIR / uf.name
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
            convert_all_xlsx(DATA_DIR)
            st.cache_data.clear()
            st.sidebar.success("Файлы сохранены и готовы к использованию")

# ------------------ Основной заголовок ------------------
st.title("Рекомендательная система для энергосбытовой компании")
st.markdown("""
Этот инструмент позволяет:
- Загрузить исходные данные об абонентах и мерах воздействия
- Обучить модель прогноза возврата долга
- Получить оптимальные рекомендации мер на выбранный месяц
- Проанализировать результаты через графики и объяснения
""")

# ------------------ Вкладки ------------------
tab1, tab2, tab3, tab4 = st.tabs(["Обучение модели", "Рекомендации", "Объяснение отдельного ЛС", "Профили кластеров"])

# ------------------ Вкладка 1: Обучение ------------------
with tab1:
    st.header("Обучение модели прогноза снижения долга")
    if st.button("Обучить модель", key="train_btn"):
        with st.spinner("Построение выборки и обучение... Это может занять много времени."):
            subprocess.run([sys.executable, "build_training_dataset.py"], check=True)
            subprocess.run([sys.executable, "train_model.py"], check=True)
        st.success("Модель обучена!")

        metrics_path = MODELS_DIR / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.subheader("Качество модели (кросс-валидация)")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", metrics.get("R2", "-"))
            col2.metric("RMSE", f"{metrics.get('RMSE', '-')} руб.")
            col3.metric("MAE", f"{metrics.get('MAE', '-')} руб.")

        try:
            model = joblib.load(MODELS_DIR / "uplift_model.pkl")
            feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
            importance = model.feature_importance(importance_type='gain')
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
            imp_df = imp_df.sort_values('importance', ascending=False).head(10)
            # Попробуем перевести названия на человеческий язык
            from config import FEATURE_HUMAN

            imp_df['feature_human'] = imp_df['feature'].apply(lambda x: FEATURE_HUMAN.get(x, (x,))[0])
            fig = px.bar(imp_df, x='importance', y='feature_human', orientation='h',
                         title="Топ-10 важных признаков")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось отобразить важность признаков: {e}")

# ------------------ Вкладка 2: Рекомендации ------------------
with tab2:
    st.header("Формирование рекомендаций")

    # Выбор месяца
    available_months = get_available_months()
    if not available_months:
        st.warning("Данные оборотной ведомости не найдены. Загрузите данные.")
    else:
        target_month = st.selectbox("Целевой месяц", available_months,
                                    format_func=lambda d: d.strftime('%Y-%m'))
        if st.button("Сформировать рекомендации", key="recommend_btn"):
            with st.spinner("Идёт вычисление признаков и оптимальное распределение мер..."):
                os.environ["TARGET_MONTH"] = target_month.strftime("%Y-%m-%d")
                result = subprocess.run([sys.executable, "recommend.py"], capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Ошибка:\n{result.stderr}")
                else:
                    st.success("Рекомендации сформированы!")

            rec_df = load_recommendations()
            if rec_df is not None:
                st.subheader("Пример рекомендаций (первые 50 строк)")
                st.dataframe(rec_df.head(50))
                st.download_button("Скачать CSV с рекомендациями", data=rec_df.to_csv(index=False),
                                   file_name=f"recommendations_{target_month.strftime('%Y-%m')}.csv")

                # Графики
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(rec_df, x="expected_return", nbins=50,
                                       title="Распределение ожидаемого возврата")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    # Использование лимитов с человеческими названиями
                    limits_summary = rec_df['action_name'].value_counts().reset_index()
                    limits_summary.columns = ['Мера', 'Использовано']
                    # Лимиты добавим (можно загрузить из data_loader)
                    from data_loader import load_all_data

                    limits_df_raw = load_all_data()['limits']
                    limits_df_raw['Мера'] = limits_df_raw['Мера'].map({
                        '04 Автодозвон ХК': 'autodial',
                        '05 E-mail ХК': 'email',
                        '06 СМС ХК': 'sms',
                        '07 Обзвон оператором ХК': 'operator_call',
                        '08 Претензия ХК': 'claim',
                        '09 Выезд к абоненту ХК': 'visit',
                        '10 Уведомление о введении ограничения ХК': 'notice_limit',
                        '11 Ограничение ХК': 'limit',
                        '12 Заявление о выдаче судебного приказа ХК': 'court_order',
                        '13 Получение судебного приказа или ИЛ ХК': 'exec_doc'
                    })
                    limits_df_raw['Мера'] = limits_df_raw['Мера'].map(ACTION_NAMES)
                    limits_df_raw = limits_df_raw.rename(columns={'Лимит': 'Лимит'})
                    limits_summary = limits_summary.merge(limits_df_raw, on='Мера', how='left')
                    limits_summary['Лимит'] = limits_summary['Лимит'].fillna(0).astype(int)
                    limits_summary['Использовано %'] = (
                            limits_summary['Использовано'] / limits_summary['Лимит'] * 100).fillna(0).round(1)
                    fig = px.bar(limits_summary, x='Мера', y='Использовано',
                                 text='Использовано', title="Использование лимитов")
                    fig.add_scatter(x=limits_summary['Мера'], y=limits_summary['Лимит'],
                                    mode='markers+lines', name='Лимит', marker=dict(color='red'))
                    st.plotly_chart(fig, use_container_width=True)

                # Средний возврат по кластерам
                if 'cluster' in rec_df.columns:
                    cluster_stats = rec_df.groupby('cluster')['expected_return'].mean().reset_index()
                    fig = px.bar(cluster_stats, x='cluster', y='expected_return',
                                 title="Средний ожидаемый возврат по кластерам")
                    st.plotly_chart(fig, use_container_width=True)

                # Scatter
                sample = rec_df.sample(min(5000, len(rec_df)))
                fig = px.scatter(sample, x='debt_current', y='expected_return', color='action_name',
                                 title="Долг vs Ожидаемый возврат (выборка)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Файл с рекомендациями не найден. Возможно, произошла ошибка.")

# ------------------ Вкладка 3: Объяснение ------------------
with tab3:
    st.header("Объяснение рекомендации для конкретного ЛС")
    ls_input = st.number_input("Введите номер лицевого счёта", min_value=1, value=69910)
    if st.button("Объяснить"):
        result = subprocess.run([sys.executable, "explain.py", str(ls_input)], capture_output=True, text=True)
        if result.returncode == 0:
            st.text(result.stdout)
        else:
            st.error("Не удалось получить объяснение. Проверьте, обучена ли модель и существуют ли данные.")

# ------------------ Вкладка 4: Профили кластеров ------------------
with tab4:
    st.header("Профили кластеров должников")
    try:
        result = subprocess.run([sys.executable, "clusters_info.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.text(result.stdout)
        else:
            st.error("Ошибка формирования описаний кластеров.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
