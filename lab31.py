import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Параметри
N = 10000
k = 20
a = 5**13
c = 2**31
z0 = 1

# Генерація випадкових чисел
def linear_congruential_generator(a, c, z0, N):
    z = z0
    numbers = []
    for _ in range(N):
        z = (a * z) % c
        x = z / c
        numbers.append(x)
    return np.array(numbers)

x = linear_congruential_generator(a, c, z0, N)

#  Побудова гістограми
plt.hist(x, bins=k, density=True, alpha=0.6, color='g', label="Empirical Histogram")
plt.title("Histogram of Generated Values")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()


# Побудова інтервалів для тестування
hist, bin_edges = np.histogram(x, bins=k, density=False)
expected_freq = [N / k] * k

# Обчислення значення χ^2
observed_freq = hist
chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

# Критичне значення χ^2
df = k - 1
critical_value = chi2.ppf(0.95, df)

M_x = np.mean(x)
D_x = np.var(x)

print(f"Математичне сподівання: {M_x}")
print(f"Дисперсія: {D_x}")

# Виведення результатів
print(f"Chi-squared Statistic: {chi2_stat:.2f}")
print(f"Critical Value (alpha=0.05): {critical_value:.2f}")
if chi2_stat < critical_value:
    print("Гіпотезу про рівномірний розподіл приймаємо.")
else:
    print("Гіпотезу про рівномірний розподіл відхиляємо.")
