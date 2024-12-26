import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from lab41 import best_coefficients

# Дані з таблиці
x = np.array([1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61])
y = np.array([312.89, 1612, 4225, 8043, 12900, 18560, 24740, 31070, 37160, 42510, 46600, 48820, 48510, 44960, 37370, 24910])

# Функція для побудови поліноміальної моделі
def fit_polynomial_model(x, y, degree):
    # Поліноміальна регресія
    coefficients = np.polyfit(x, y, degree)
    model = np.poly1d(coefficients)
    y_pred = model(x)

    # Критерій найменших квадратів (MSE)
    mse = mean_squared_error(y, y_pred)

    return coefficients, y_pred, mse

# Аналіз моделей з різними степенями полінома
# Аналіз моделей різної складності
results = {}
for degree in range(1, 6):  # Ступені від 1 до 5
    coefficients, y_pred, mse = fit_polynomial_model(x, y, degree)
    results[degree] = (coefficients, y_pred, mse)

# Вибір найкращої моделі
best_degree = min(results, key=lambda d: results[d][2])
best_coefficients, best_y_pred, best_mse = results[best_degree]

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Дані спостережень')
plt.plot(x, best_y_pred, color='red', label=f'Поліном ступеня {best_degree}')
plt.title('Ідентифікація обєкта за методом найменших квадраті')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Виведення таблиці результатів
print("Ступінь полінома:", best_degree)
print("Критерій найменших квадратів (MSE):", best_mse)
print("Коефіцієнти моделі:", best_coefficients)
print("Таблиця значень:")
print("x\tСпостережене y\tМодельне y")
for xi, yi, yi_pred in zip(x, y, best_y_pred):
    print(f"{xi}\t{yi:.2f}\t\t{yi_pred:.2f}")
