import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Загрузка данных из файла
file_path = '20240722 РостАгро, тестовое задание (датасет).xlsx'  # Убедитесь, что путь к файлу указан правильно
data = pd.read_excel(file_path, sheet_name='Y', header=4)

# Загрузка данных о ценах на товары
commodity_data = pd.DataFrame({
    'Year': pd.to_datetime(data.iloc[:, 0], format='%Y', errors='coerce'),
    'Soybeans': data.iloc[:, 8],
    'Sunflower Oil': data.iloc[:, 9],
    'Wheat': data.iloc[:, 12],
    'Phosphate Rock': data.iloc[:, 13]
})

commodity_data.set_index('Year', inplace=True)
commodity_data = commodity_data.apply(pd.to_numeric, errors='coerce')
commodity_data.dropna(inplace=True)

# Определение дополнительных переменных
additional_columns = {
    'Inflation_USA': 15,
    'Population_total': 16,
    'Precipitation': 17,
    'Cereal_yield': 18,
    'Agri_land': 19,
    'Cereal_production': 20,
    'Climate_temperature_change': 21
}

# Извлечение и обработка данных для экзогенных переменных
exog_data = pd.DataFrame({
    'Year': pd.to_datetime(data.iloc[:, 0], format='%Y', errors='coerce')
})

for var_name, col_index in additional_columns.items():
    exog_data[var_name] = data.iloc[:, col_index]

exog_data.set_index('Year', inplace=True)
exog_data = exog_data.apply(pd.to_numeric, errors='coerce')
exog_data.fillna(exog_data.mean(), inplace=True)

# Перебор параметров модели ARIMA для каждого товара с использованием auto_arima
best_models = {}

for commodity in ['Soybeans', 'Sunflower Oil', 'Wheat', 'Phosphate Rock']:
    commodity_series = commodity_data[commodity]
    exog_vars = exog_data.loc[commodity_series.index]

    # Использование auto_arima для подбора наилучших параметров (p, d, q)
    model = auto_arima(commodity_series, exogenous=exog_vars, seasonal=False, trace=True, error_action='ignore',
                       suppress_warnings=True)

    best_models[commodity] = model

    # Вывод лучших параметров модели
    print(f"Best model for {commodity}: {model.summary()}")

# Прогнозирование и визуализация результатов для лучшей модели
plt.figure(figsize=(14, 8))

for commodity, model in best_models.items():
    commodity_series = commodity_data[commodity]
    exog_vars = exog_data.loc[commodity_series.index]

    # Прогнозирование на период вперед
    forecast_period = 2030 - 2023 + 1
    forecast_exog = pd.concat([exog_vars.iloc[-1:]] * forecast_period, ignore_index=True)
    forecast_exog.index = pd.date_range(start=commodity_series.index[-1] + pd.DateOffset(years=1),
                                        periods=forecast_period, freq='YS')

    forecast = model.predict(n_periods=forecast_period, exogenous=forecast_exog)
    forecast_index = forecast_exog.index

    # Вывод прогнозируемых цен
    print(f"\nForecast for {commodity} (2024-2030):")
    for date, price in zip(forecast_index, forecast):
        print(f"{date.year}: {price:.2f} $/mt")

    # Визуализация реальных данных и прогнозов
    plt.plot(commodity_data.index, commodity_data[commodity],
             label=f'Real {commodity} (until {commodity_series.index[-1].year})')
    plt.plot(forecast_index, forecast, linestyle='--', label=f'Forecast {commodity} (2024-2030)')

plt.xlim(pd.Timestamp('1960-01-01'), pd.Timestamp('2030-12-31'))
plt.title('Historical and Forecasted Prices of Commodities using Optimized ARIMA')
plt.xlabel('Year')
plt.ylabel('Price ($/mt)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
