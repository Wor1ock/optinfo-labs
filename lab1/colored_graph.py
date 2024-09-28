import matplotlib.pyplot as plt
import numpy as np


def K(f_arr, ksi, x_arr, alpha):
    h = (b - a) / m
    fl = complex(0, 0)
    for x_i, f_i in zip(x_arr, f_arr):
        fl += x_i ** (alpha * ksi - 1) * f_i * h
    return fl


def f(x, beta):
    return np.exp(complex(0, 1) * beta * x)


def calculate_F(f_arr, x_arr, ksi_arr, alpha):
    g_arr = np.zeros(m, dtype=complex)
    for i, ksi in enumerate(ksi_arr):
        g_arr[i] = K(f_arr, ksi, x_arr, alpha)
    return g_arr


def plot_color_schemes(x_arr, ksi_arr, matrix, title):
    plt.figure(figsize=(8, 6))

    plt.imshow(matrix, aspect='auto', cmap='inferno', extent=(x_arr.min(), x_arr.max(), ksi_arr.min(), ksi_arr.max()))
    plt.colorbar(label=title)
    plt.title(f'{title}')
    plt.xlabel('x')
    plt.ylabel('ξ')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def experiment_color_scheme(x_arr, ksi_arr, alpha, beta):
    amplitude_matrix = np.zeros((len(ksi_arr), len(x_arr)))
    phase_matrix = np.zeros((len(ksi_arr), len(x_arr)))

    f_x = f(x_arr, beta)

    for i, ksi in enumerate(ksi_arr):
        kernel_value = K(f_x, ksi, x_arr, alpha)
        amplitude_matrix[i, :] = np.abs(kernel_value)
        phase_matrix[i, :] = np.angle(kernel_value)

    plot_color_schemes(x_arr, ksi_arr, amplitude_matrix, 'Amplitude of K(x, ξ)')
    plot_color_schemes(x_arr, ksi_arr, phase_matrix, 'Phase of K(x, ξ)')


if __name__ == '__main__':
    m, n = 1000, 1000
    a, b = 1, 5
    p, q = 0, 3
    alpha, beta = 1, 1 / 10

    x_arr = np.linspace(a, b, n)
    ksi_arr = np.linspace(p, q, m)

    experiment_color_scheme(x_arr, ksi_arr, alpha, beta)