import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Загрузите данные из вашего CSV-файла
data = pd.read_csv('output.csv')

# Преобразуйте дату в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Разделите данные на признаки (X) и целевую переменную (y)
X = data[['date', 'region']]
y = data['value']

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте и обучите модель CatBoost с настроенными гиперпараметрами
model = CatBoostRegressor(iterations=100, depth=15, learning_rate=0.5, loss_function='RMSE', border_count=128, l2_leaf_reg=3.0)
model.fit(X_train, y_train, cat_features=['region'])

# Сделайте прогнозы
y_pred = model.predict(X_test)

# Оцените качество модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
