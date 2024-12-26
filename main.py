import numpy as np
import matplotlib.pyplot as plt

n = 6  # Число гармонік
omega_gr = 2100  # Гранична частота
N = 256  # Кількість дискретних відліків

k_zap = 5  # Коефіцієнт вибірки
f_gr = omega_gr / (2 * np.pi)  # Переведення частоти в Гц

dt = 1 / (k_zap * f_gr)  # Крок дискретизації
t = np.linspace(0, (N - 1) * dt, N)

def generate_signal(omega_gr, n, t):
    A = np.random.uniform(0, 1, n)  # Амплітуди
    phi = np.random.uniform(0, 2 * np.pi, n)  # Фази
    omega = np.linspace(0, omega_gr, n)  # Частоти

    x_t = np.zeros(N)
    for i in range(n):
        x_t += A[i] * np.sin(omega[i] * t + phi[i])
    return x_t

x_t = generate_signal(omega_gr, n, t)
y_t = generate_signal(omega_gr, n, t)

M_x = np.mean(x_t)
M_y = np.mean(y_t)
D_x = np.var(x_t)

print(f"Математичне сподівання: {M_x}")
print(f"Дисперсія: {D_x}")

plt.figure(figsize=(10, 6))
plt.plot(t, x_t, label="x(t)")
plt.title("Генерація випадкового сигналу")
plt.xlabel("Час (с)")
plt.ylabel("Амплітуда")
plt.grid()
plt.legend()
plt.show()

# Обчислення автокореляційної функції
def autocorrelation(x, tau_max):
    N = len(x)
    result = []
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            corr = np.sum((x[:N - tau] - M_x) * (x[tau:] - M_x)) / (N - 1)
        else:
            corr = np.sum((x[-tau:] - M_x) * (x[:N + tau] - M_x)) / (N - 1)
        result.append(corr)
    return np.array(result)

tau_max = 50 # Максимальна затримка
R_xx = autocorrelation(x_t, tau_max)
taus = np.arange(-tau_max, tau_max + 1) * dt  # Масштаб τ у часі

# Графік результату
plt.figure(figsize=(10, 6))
plt.plot(taus, R_xx, label='Автокореляційна функція $R_{xx}(\\tau)$', color='b')
plt.title('Автокореляційна функція сигналу $R_{xx}(\\tau)$')
plt.xlabel('Затримка $\\tau$, сек')
plt.ylabel('$R_{xx}(\\tau)$')
plt.grid()
plt.legend()
plt.show()

def cross_correlation(x, y, tau_max):
    N = len(x)
    result = []
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            corr = np.sum((x[:N - tau] - M_x) * (y[tau:] - M_y)) / (N - 1)
        else:
            corr = np.sum((x[-tau:] - M_x) * (y[:N + tau] - M_y)) / (N - 1)
        result.append(corr)
    return np.array(result)


R_xy = cross_correlation(x_t, y_t, tau_max)

plt.figure(figsize=(10, 6))
plt.plot(taus, R_xy, label='Взаємна кореляційна функція $R_{xy}(\\tau)$', color='g')
plt.title('Взаємна кореляційна функція $R_{xy}(\\tau)$')
plt.xlabel('Затримка $\\tau$, сек')
plt.ylabel('$R_{xy}(\\tau)$')
plt.grid()
plt.legend()
plt.show()


