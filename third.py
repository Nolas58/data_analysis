import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns


# Загрузка и подготовка данных
file_path = '20240722 РостАгро, тестовое задание (датасет).xlsx'
data = pd.read_excel(file_path, sheet_name='Y', header=None)

# Определение правильных столбцов для каждого товара
commodity_data = pd.DataFrame({
    'Year': pd.to_datetime(data.iloc[4:, 0], format='%Y', errors='coerce'),
    'Soybeans': data.iloc[4:, 8],
    'Sunflower Oil': data.iloc[4:, 9],
    'Wheat': data.iloc[4:, 12],
    'Phosphate Rock': data.iloc[4:, 13]
})

commodity_data.set_index('Year', inplace=True)
commodity_data = commodity_data.apply(pd.to_numeric, errors='coerce')
commodity_data.dropna(inplace=True)


# Устанавливаем частоту индекса, если это возможно
try:
    inferred_freq = pd.infer_freq(commodity_data.index)
    commodity_data.index.freq = inferred_freq
except ValueError as e:
    print(f"Warning: {e}")


# Прогнозирование на 2024-2030 годы с упрощенными параметрами SARIMA
forecasted_data = {}
forecast_period = 2030 - 2023 + 1
forecast_index = pd.date_range(start='2024', periods=forecast_period, freq='YS')

# Упрощенные параметры SARIMA
simplified_params = (1, 0, 1, 0, 1, 1, 12)

for column in ['Soybeans', 'Sunflower Oil', 'Wheat', 'Phosphate Rock']:
    model_final = SARIMAX(commodity_data[column], order=simplified_params[:3], seasonal_order=simplified_params[3:])
    model_fit_final = model_final.fit(disp=False)
    forecast = model_fit_final.get_forecast(steps=forecast_period)
    forecast_values = forecast.predicted_mean
    forecasted_data[column] = forecast_values

forecast_df_all = pd.DataFrame(forecasted_data, index=forecast_index)


# Визуализация временных рядов с реальными и прогнозируемыми данными с совпадением цветов
plt.figure(figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(['Soybeans', 'Sunflower Oil', 'Wheat', 'Phosphate Rock'])))
for i, column in enumerate(['Soybeans', 'Sunflower Oil', 'Wheat', 'Phosphate Rock']):
    plt.plot(commodity_data.index, commodity_data[column], color=colors[i], label=f'Реальные данные: {column}')
    plt.plot(forecast_df_all.index, forecast_df_all[column], linestyle='--', marker='o', color=colors[i], label=f'Прогнозируемые данные: {column}')

plt.title('Прогнозирование цен на товары (упрощенная SARIMA) (2024-2030)')
plt.xlabel('Год')
plt.ylabel('Цена ($/mt)')
plt.legend()
plt.grid(True)
plt.xticks(pd.date_range(start=commodity_data.index.min(), end='2031-01-01', freq='YS'), rotation=45, fontsize=8, ha='right')
plt.xlim(left=commodity_data.index.min(), right=forecast_df_all.index.max())

plt.show()
# Печать прогнозируемых данных
print(forecast_df_all)

# Корреляционная матрица
correlation_matrix = commodity_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица цен на товары')
plt.show()

# Графики распределения цен
plt.figure(figsize=(14, 10))
for i, column in enumerate(commodity_data.columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(commodity_data[column], kde=True)
    plt.title(f'Распределение цен на {column}')
    plt.xlabel('Цена ($/mt)')
    plt.ylabel('Частота')

plt.tight_layout()
plt.show()
