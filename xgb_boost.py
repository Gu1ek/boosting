import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Загрузите данные из вашего CSV-файла
data = pd.read_csv('output.csv')

# Преобразуйте дату в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Создайте дополнительные фичи на основе даты
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Разделите данные на признаки (X) и целевую переменную (y)
X = data[['year', 'month', 'region']]
y = data['value']

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте DMatrix для XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Определите параметры модели
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,  # Измените на оптимальное значение
    'eta': 0.1,  # Измените на оптимальное значение
    'subsample': 0.8,  # Измените на оптимальное значение
    'colsample_bytree': 0.8,  # Добавьте параметр
}

# Обучите модель
num_round = 1000  # Измените на оптимальное значение
bst = xgb.train(params, dtrain, num_round)

# Сделайте прогноз
y_pred = bst.predict(dtest)

# Оцените качество модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
