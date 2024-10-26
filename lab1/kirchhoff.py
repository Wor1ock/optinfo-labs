import numpy as np
import matplotlib.pyplot as plt

# Физические параметры
sigma = 0.5 * 1e-3  # Радиус гауссова пучка, м
z = 1  # Дальность распространения, м
lambda_wave = 1e-6  # Длина волны, м
k = 2 * np.pi / lambda_wave  # Волновое число, м^-1

# Параметры области
a1, b1 = -5e-3, 5e-3  # Границы по x, м
a2, b2 = -5e-3, 5e-3  # Границы по y, м


# Функция гауссова пучка
def gaussian_beam(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


# Расстояние между точками (x, y) и (ksi, eta)
def distance_R(x, y, ksi, eta, z):
    return np.sqrt((x - ksi) ** 2 + (y - eta) ** 2 + z ** 2)


# Интеграл Кирхгоффа
def kirchhoff_integral(x_arr, y_arr, ksi_arr, eta_arr, sigma, z, k):
    result = np.zeros((len(ksi_arr), len(eta_arr)), dtype=complex)

    dx = (b1 - a1) / len(x_arr)
    dy = (b2 - a2) / len(y_arr)

    coeff = -1j * k / (2 * np.pi)

    for i, ksi in enumerate(ksi_arr):
        for j, eta in enumerate(eta_arr):
            integral_sum = 0
            for x in x_arr:
                for y in y_arr:
                    R = distance_R(x, y, ksi, eta, z)
                    f_xy = gaussian_beam(x, y, sigma)
                    integral_sum += f_xy * np.exp(1j * k * R) * (z / R) / R * dx * dy
            result[i, j] = coeff * integral_sum
        print(f"Итерация {i + 1} из {len(ksi_arr)}")
    return result



def experiment_kirchhoff():
    n = 32
    x_arr = np.linspace(a1, b1, n)
    y_arr = np.linspace(a2, b2, n)
    ksi_arr = np.linspace(a1, b1, n)
    eta_arr = np.linspace(a2, b2, n)

    F_array = kirchhoff_integral(x_arr, y_arr, ksi_arr, eta_arr, sigma, z, k)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(ksi_arr, eta_arr, np.abs(F_array), cmap='viridis')
    plt.title("Амплитуда выходного сигнала")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.contourf(ksi_arr, eta_arr, np.angle(F_array), cmap='viridis')
    plt.title("Фаза выходного сигнала")
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    experiment_kirchhoff()
