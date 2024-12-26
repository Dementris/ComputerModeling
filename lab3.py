import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Параметри
N = 10000  # Кількість випадкових чисел
lmbda = 0.2  # Параметр розподілу (λ)
k = 20  # Кількість інтервалів для гістограми

# Генерація випадкових чисел
xi = np.random.uniform(0, 1, N)  # Рівномірно розподілені ξi
x = -1 / lmbda * np.log(xi)      # Генерація за формулою

# Побудова гістограми
plt.hist(x, bins=k, density=True, alpha=0.6, color='g', label="Empirical Histogram")

# Теоретична функція розподілу
x_vals = np.linspace(0, max(x), 1000)
pdf = lmbda * np.exp(-lmbda * x_vals)
plt.plot(x_vals, pdf, 'r', lw=2, label="Theoretical")

plt.title("Histogram of Generated Values and Theoretical Exponential")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()

# Критерій згоди χ^2
hist, bin_edges = np.histogram(x, bins=k, density=False)

# Очікувані частоти за теоретичним розподілом
expected_freq = []
for i in range(len(bin_edges) - 1):
    cdf_upper = 1 - np.exp(-lmbda * bin_edges[i + 1])
    cdf_lower = 1 - np.exp(-lmbda * bin_edges[i])
    expected_prob = cdf_upper - cdf_lower
    expected_freq.append(expected_prob * N)

# Обчислимо значення χ^2
observed_freq = hist
chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

df = k - 2
critical_value = chi2.ppf(0.95, df)

M_x = np.mean(x)
D_x = np.var(x)

print(f"Математичне сподівання: {M_x}")
print(f"Дисперсія: {D_x}")


# Виведення результатів
print(f"Chi-squared Statistic: {chi2_stat:.2f}")
print(f"Critical Value (alpha=0.05): {critical_value:.2f}")
if chi2_stat < critical_value:
    print("Гіпотезу про експоненційний розподіл приймаємо.")
else:
    print("Гіпотезу про експоненційний розподіл відхиляємо.")
