import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Дані з таблиці для функції y = b0 + b1 * (1/x) + b2 * (1/x^2) + ...
x = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5])
y = np.array([14, 18.222, 18, 17.216, 16.444, 15.778, 15.219, 14.749, 14.352, 14.014, 13.722, 13.469, 13.248, 13.052, 12.879, 12.724])

# Перетворення для вигляду функції: додаємо похідні від 1/x до 1/x^n
def transform_features(x, degree):
    return np.array([1 / (x ** i) for i in range(1, degree + 1)]).T

# Функція для побудови моделі

def fit_inverse_polynomial_model(x, y, degree):
    X_transformed = transform_features(x, degree)
    X_augmented = np.column_stack((np.ones(x.shape[0]), X_transformed))  # Додаємо константу для b0
    coefficients = np.linalg.lstsq(X_augmented, y, rcond=None)[0]  # Розв'язок методу найменших квадратів
    y_pred = X_augmented @ coefficients
    mse = mean_squared_error(y, y_pred)
    return coefficients, y_pred, mse

# Аналіз моделей різної складності
results = {}
for degree in range(1, 6):  # Ступені від 1 до 5
    coefficients, y_pred, mse = fit_inverse_polynomial_model(x, y, degree)
    results[degree] = (coefficients, y_pred, mse)

# Вибір найкращої моделі
best_degree = min(results, key=lambda d: results[d][2])
best_coefficients, best_y_pred, best_mse = results[best_degree]

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Дані спостережень')
plt.plot(x, best_y_pred, color='red', label=f'Найкраща модель ступеня {best_degree}')
plt.title('Ідентифікація обєкта: функція з оберненими ступенями')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Виведення таблиці результатів
print(f"Найкраща модель: ступінь = {best_degree}")
print(f"Критерій найменших квадратів (MSE): {best_mse:.20f}")
print("Коефіцієнти моделі:", best_coefficients)
print("Таблиця значень:")
print("x\tСпостережене y\tМодельне y")
for xi, yi, yi_pred in zip(x, y, best_y_pred):
    print(f"{xi:.1f}\t{yi:.3f}\t\t{yi_pred:.3f}")
