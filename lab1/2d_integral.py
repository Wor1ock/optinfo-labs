import enum

import matplotlib.pyplot as plt
import numpy as np


class Title(enum.Enum):
    inputA = "График амплитуды входного сигнала"
    outputA = "График амплитуды выходного сигнала"
    inputF = "График фазы входного сигнала"
    outputF = "График фазы выходного сигнала"


def K2D(f_arr, ksi, eta, x_arr, y_arr, alpha):
    h = (b - a) * (d - c) / (m * n)
    F_lm = complex(0, 0)
    for x_i, f_i in zip(x_arr, f_arr):
        for y_j, f_j in zip(y_arr, f_i):
            x_complex = complex(x_i) if x_i < 0 else x_i
            y_complex = complex(y_j) if y_j < 0 else y_j

            F_lm += np.power(x_complex, alpha * ksi - 1) * np.power(y_complex, alpha * eta - 1) * f_j * h
    return F_lm


# Входной сигнал f(x_k, y_k) = f(x) * f(y)
def f2D(x, y, beta):
    return np.exp(complex(0, 1) * beta * (x + y))


# Основная функция для расчета F(ξ_l, η_m)
def calculate_F2D(f_arr, x_arr, y_arr, ksi_arr, eta_arr, alpha):
    G_arr = np.zeros((m, n), dtype=complex)
    for i, ksi in enumerate(ksi_arr):
        for j, eta in enumerate(eta_arr):
            G_arr[i, j] = K2D(f_arr, ksi, eta, x_arr, y_arr, alpha)
        print(f"Итерация {i + 1} из {m}")
    return G_arr


def plot_graphs_2D(x_arr, y_arr, abs_arr, angle_arr, alpha, beta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].contourf(x_arr, y_arr, abs_arr, cmap='viridis')
    axes[0].set_title(f"График амплитуды\nα={alpha}, β={beta}", color='r')
    axes[0].set_xlabel("ξ")
    axes[0].set_ylabel("η")
    axes[0].grid(True)
    fig.colorbar(axes[0].collections[0], ax=axes[0])

    axes[1].contourf(x_arr, y_arr, angle_arr, cmap='viridis')
    axes[1].set_title(f"График фазы\nα={alpha}, β={beta}", color='r')
    axes[1].set_xlabel("ξ")
    axes[1].set_ylabel("η")
    axes[1].grid(True)
    fig.colorbar(axes[1].collections[0], ax=axes[1])

    plt.suptitle("Результаты преобразования", fontsize=14)
    plt.tight_layout()
    plt.show()


def experiment_2D(x_arr, y_arr, ksi_arr, eta_arr, alpha, beta):
    f_xy = f2D(x_arr[:, None], y_arr, beta)
    G_array = calculate_F2D(f_xy, x_arr, y_arr, ksi_arr, eta_arr, alpha)

    plot_graphs_2D(ksi_arr, eta_arr, np.abs(G_array), np.angle(G_array), alpha, beta)


if __name__ == '__main__':
    m, n = 32, 32
    a, b = -3, 3
    c, d = -3, 3
    p, q = -3, 3
    r, s = -3, 3
    alpha, beta = 1, 1 / 10

    x_arr = np.linspace(a, b, n)
    y_arr = np.linspace(c, d, n)
    ksi_arr = np.linspace(p, q, m)
    eta_arr = np.linspace(r, s, m)

    experiment_2D(x_arr, y_arr, ksi_arr, eta_arr, alpha, beta)
