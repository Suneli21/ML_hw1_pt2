import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import phik
import os

#============ очистка ==========
def get_basic_info(df):
    unique_counts = df.nunique()
    return [
        f'Колонки с пропусками: {', '.join(df.columns.to_series()[df.isna().any()].values)}',
        f'Колонки с дубликатами: {', '.join(unique_counts[unique_counts != df.shape[0]].index)}',
        'В данных есть явные дубликаты (строчки, где все признаки принимают одинаковые значения)' if df.drop_duplicates().shape[0] != df.shape[0] else 'Явных дубликатов (строчек, где все признаки принимают одинаковые значения) нет'
    ]

def get_num_cat_describe(df):
    return [df.describe(), df.describe(include='object')]

def clean_numeric_col(row):
  cols = {'mileage':['kmpl', 'km/kg'], 'engine': ['CC'], 'max_power': 'bhp'}
  for key in cols.keys():
    if pd.notna(row[key]):
      for pattern in cols[key]:
        row[key] = str(row[key]).replace(pattern, '').strip()
  return row

def clean_duplicates(df):
    df.drop_duplicates(keep='first', subset=[col for col in df.columns if col != 'selling_price'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def extract_torque_values(torque_str):
    if pd.isna(torque_str):
        return [None, None]

    torque_str = str(torque_str).lower().strip()
    # обработка одного из шаблонов: 400 Nm /2000 rpm
    if '/' in torque_str:
        parts = torque_str.split('/')
        if len(parts) == 2:
            torque_part = parts[0].strip()
            rpm_part = parts[1].strip()
            # извлекаем числа
            torque_match = re.search(r'(\d+\.?\d*)', torque_part)
            rpm_match = re.search(r'(\d+)', rpm_part)
            if torque_match and rpm_match:
                torque = float(torque_match.group(1))
                max_rpm = int(rpm_match.group(1))
                if 'kg' in torque_part:
                    torque *= 9.80665
                return [torque, max_rpm]

    # обработка случаев по шаблону 210 / 1900
    if re.search(r'\d+\s*/\s*\d+', torque_str):
        parts = re.split(r'\s*/\s*', torque_str)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return [float(parts[0]), int(parts[1])]

   # ищется шаблон вида: число, пробельный символ (если есть), kgm или nm, пробельный символ (если есть), @ или at, пробельный символ (если есть), число или диапазон, может включать точку, тильду, плюс, дефис,
   # пробельный символ (если есть), rpm (находит его но не включает в результат)
    pattern = r'(\d+\.?\d*)\s*(kgm?|nm?)?\s*(?:@|at)?\s*([\d\s\.,~+\-]+)(?:\s*rpm)'

    match = re.search(pattern, torque_str)

    # в случае, если ничего не найдено
    if not match:
        return [None, None]

    torque_value = float(match.group(1))
    unit = match.group(2) if match.group(2) else ''
    rpm_info = match.group(3).strip()

    if unit and 'kg' in unit:
        torque_value *= 9.80665
    max_rpm = extract_max_rpm(rpm_info)

    return [torque_value, max_rpm]

def extract_max_rpm(rpm_info):
    #очистка от лишних символов
    rpm_info = rpm_info.replace(',', '').replace(' ', '')

    #обработка диапазона - берётся среднее из обеих частей
    if '-' in rpm_info or '~' in rpm_info:
        separator = '-' if '-' in rpm_info else '~'
        parts = rpm_info.split(separator)
        if len(parts) == 2:
            try:
                rpm1 = int(parts[0])
                rpm2 = int(parts[1])
                return (rpm1 + rpm2) // 2
            except ValueError:
                return None

    if '+' in rpm_info:
        base_rpm = rpm_info.replace('+', '')
        if base_rpm.isdigit():
            return int(base_rpm)

    if rpm_info.isdigit():
        return int(rpm_info)
    match = re.search(r'(\d+)', rpm_info)
    if match:
        return int(match.group(1))

    return None

def process_torque_column(df, torque_column='torque'):
    results = df[torque_column].apply(extract_torque_values)
    df['torque'] = results.apply(lambda x: x[0] if x[0] is not None else np.nan)
    df['max_torque_rpm'] = results.apply(lambda x: x[1] if x[1] is not None else np.nan)
    return df


def object_to_numeric_and_fillna(df, fill_nan: bool=True, save_fill_values: bool=True, use_preloaded_fill_values: bool=False):
    df = df.apply(clean_numeric_col, axis=1)
    df[['mileage', 'engine', 'max_power']] = df[['mileage', 'engine', 'max_power']].astype('float')
    df = process_torque_column(df)
    fill_values = []
    preloaded_fill_values = pd.read_csv('data/fill_values.csv', index_col='col') if os.path.exists('data/fill_values.csv') else None
    if fill_nan:
        for col in df.columns:
            if df[col].dtype != 'O':
                fill_value = df[col].median() if not use_preloaded_fill_values else preloaded_fill_values.loc[col].value
                df[col] = df[col].fillna(fill_value)
                if save_fill_values:
                    fill_values.append({'col': col, 'value': fill_value})
        if save_fill_values:
            fill_values = pd.DataFrame(fill_values)
            fill_values.to_csv(r'data/fill_values.csv', index=False)

    df[['engine', 'seats']] = df[['engine', 'seats']].astype('int')
    df['name'] = df['name'].apply(lambda x: x.lower().split()[0])
    return df


#============ визуализации и статистика ==========

def corr_graph(df, method: str = 'pearson'):
    if method in ['pearson', 'spearman']:
        numeric_cols = [col for col in df.columns if df[col].dtype != 'O']
        corrs = df[numeric_cols].corr(method=method)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corrs, cmap='viridis', annot=True, fmt='.2f', ax=ax)
        plt.title(f'Корреляционная матрица ({method})', fontsize=16)
        plt.tight_layout()
        return fig
    else:
        return None


def phik_corr(df):
    phik_corrs = df.phik_matrix()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(phik_corrs, cmap='viridis', ax=ax)
    plt.title('Phi-k корреляционная матрица', fontsize=16)
    plt.tight_layout()
    return fig


def brand_plot(df):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.stripplot(data=df, x='name', y='selling_price', jitter=True, alpha=0.5, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    plt.title('Распределение цен по брендам', fontsize=16)
    plt.tight_layout()
    return fig


def get_class(x):
    if x < 200000:
        return 'low-budget'
    elif x < 500000:
        return 'lower econom'
    elif x < 1000000:
        return 'upper econom'
    else:
        return 'business'


def class_plot(df):
    brands_medians = df[['name', 'selling_price']].groupby('name').median().sort_values(by='selling_price')
    brands_medians['class'] = brands_medians['selling_price'].apply(get_class)
    brand_df = df.copy()
    brand_df['class'] = brand_df['name'].apply(lambda x: brands_medians['class'][x])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=brand_df, x='class', hue='class', ax=ax, legend=False)
    plt.title('Распределение по классам автомобилей', fontsize=16)
    plt.tight_layout()
    return fig


def pairplots(df):
    numeric_cols = [col for col in df.columns if df[col].dtype != 'O']
    if len(numeric_cols) > 8:  # Ограничиваем для производительности
        numeric_cols = numeric_cols[:8]

    pair_grid = sns.pairplot(df[numeric_cols])
    pair_grid.figure.suptitle('Pairplot числовых признаков', y=1.02, fontsize=16)
    plt.tight_layout()
    return pair_grid.figure