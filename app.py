import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from edaUtils import object_to_numeric_and_fillna, get_basic_info, get_num_cat_describe
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Предсказание цены автомобиля", page_icon="📊", layout="wide")
st.title("Предсказание цены автомобиля")
st.markdown("""
Загрузите CSV-файл, чтобы получить предсказания Ridge-регрессии и визуализацию весов модели.
В данном проекте используется Ridge с alpha=1 и предобученными весами.
""")


@st.cache_data
def get_preprocessed_data(df):
    with open(r'models/encoderScaler.pkl', 'rb') as file:
        ohe_std = pickle.load(file)
    df = ohe_std.transform(df)
    return df

@st.cache_resource
def load_cached_model():
    try:
        with open(r'models/bestModel.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None


model = load_cached_model()

with st.sidebar:
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=['csv'])
    st.divider()
    st.subheader("Настройки")
    n_samples = st.slider("Количество строк для отображения", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[0] == 0:
            st.error("Файл пустой. Загрузите файл с данными.")
        else:
            basic_info = get_basic_info(df)
            for i in basic_info:
                st.write(f"• {i}")

            with st.expander("Типы данных и количество значений"):
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.code(info_str, language='text')

            with st.expander("Статистическое описание данных"):
                num_describe, cat_describe = get_num_cat_describe(df)
                st.subheader("Числовые признаки")
                st.dataframe(num_describe, width='stretch')
                st.subheader("Категориальные признаки")
                st.dataframe(cat_describe, width='stretch')
                st.markdown("---")
            with st.expander(f"Случайные {n_samples} строк из данных"):
                if df.shape[0] >= n_samples:
                    random_sample = df.sample(n=min(n_samples, len(df)), random_state=42)
                else:
                    random_sample = df.copy()

                st.dataframe(random_sample, width='stretch')

            df_cleaned = object_to_numeric_and_fillna(df, save_fill_values=False,use_preloaded_fill_values=True)
            with st.expander(f"Случайные {n_samples} строк из очищенных данных"):
                if df_cleaned.shape[0] >= n_samples:
                    random_sample = df_cleaned.sample(n=min(n_samples, len(df)), random_state=42)
                else:
                    random_sample = df_cleaned.copy()

                st.dataframe(random_sample, width='stretch')

            X_test = df_cleaned.drop(columns='selling_price')
            X_test['seats'] = X_test['seats'].apply(lambda x: str(x))
            y_test = np.log(df_cleaned['selling_price'])
            X_test = get_preprocessed_data(X_test)
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) != X_test.columns.shape[0]:
                st.error("В данных не все признаки числовые.")
            else:
                preds = model.predict(X_test)
                coefs = model.coef_
                st.subheader("Метрики качества модели")
                st.metric("R2-score", round(r2_score(y_test, preds), 4))
                st.metric("MSE", round(MSE(y_test, preds), 4))
                with st.expander(f"Случайные {n_samples} строк из очищенного датасета + предсказания"):
                    df_preds = df_cleaned.copy()
                    df_preds = df_preds.drop(columns='selling_price')
                    df_preds['prediction'] = np.exp(preds) # возвращаем в изначальную шкалу
                    if df_preds.shape[0] >= n_samples:
                        random_sample = df_preds.sample(n=min(n_samples, len(df)), random_state=42)
                    else:
                        random_sample = df_preds.copy()
                    st.dataframe(random_sample, width='stretch')
                    df_preds_csv = df_preds.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Скачать очищенные данные + предсказания",data=df_preds_csv,file_name='df_preds.csv',mime='text/csv')
                st.divider()
                st.subheader("Визуализация весов модели")
                weights_df = pd.DataFrame({'feature': X_test.columns,'weight': coefs}).sort_values('weight', ascending=False)

                st.markdown("**Таблица весов:**")
                st.dataframe(weights_df[['feature', 'weight']].reset_index(drop=True),width='stretch',height=300)

                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8))
                colors = ['red' if x < 0 else 'green' for x in weights_df['weight']]
                bars = ax2.barh(weights_df['feature'], weights_df['weight'], color=colors)
                ax2.set_xlabel('Значение веса')
                ax2.set_title('Веса признаков модели Ridge')
                ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

                st.subheader("Детальная статистика весов")
                st.metric("Количество признаков", len(coefs))
                st.metric("Максимальный вес", f"{weights_df['weight'].max():.3f}")
                st.metric("Минимальный вес", f"{weights_df['weight'].min():.3f}")
                st.metric("Средний вес", f"{weights_df['weight'].mean():.3f}")
                st.text(f"Модель занулила веса:{weights_df[weights_df['weight']==0]['feature'] if any(weights_df['weight']==0) else 'никакие'}")
                top_weights = weights_df.iloc[weights_df['weight'].abs().argsort()[::-1][:5]]
                st.text(f"Самые важные признаки, судя по весам модели: {', '.join(top_weights['feature'].tolist())}")
                st.text("Довольно интересно, что в веса попали определённые колонки из OHE-закодированных категориальных, как, например, name_chevrolet.")
                fig2, ax2, = plt.subplots(figsize=(12, 12))
                colors = ['red' if x < 0 else 'blue' for x in weights_df['weight']]
                bars = ax2.barh(weights_df['feature'], weights_df['weight'], color=colors)
                ax2.set_xlabel('Значение веса')
                ax2.set_title('Веса признаков модели Ridge')
                ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                st.pyplot(fig2)
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                ax3.hist(coefs, bins=20, edgecolor='black', alpha=0.7)
                ax3.set_xlabel('Значение веса')
                ax3.set_ylabel('Частота')
                ax3.set_title('Распределение весов модели')
                ax3.axvline(x=0, color='red', linestyle='--', label='Нулевой вес')
                ax3.legend()
                st.pyplot(fig3)


    except pd.errors.EmptyDataError:
        st.error("Файл не содержит данных. Загрузите корректный CSV-файл.")
    except Exception as e:
        st.error(f"Ошибка при обработке файла: {e}")

else:
    st.info("👈 Пожалуйста, загрузите CSV-файл через панель слева")
    st.markdown("""
        **Ridge регрессия:** alpha= 1.0 (сила регуляризации, коэффициент был подобран при выполнении ДЗ 1  части 1)
        
        **Визуализация показывает:**
        - Статистические характеристики весов
        - Значения весов каждого признака
        - Распределение признаков
        """)

