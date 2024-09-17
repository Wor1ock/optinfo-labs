import enum
import matplotlib.pyplot as plt
import numpy as np


class Title(enum.Enum):
    inputA = "График амплитуды входного сигнала"
    outputA = "График амплитуды выходного сигнала"
    inputF = "График фазы входного сигнала"
    outputF = "График фазы выходного сигнала"


# Функция ядра K(ξ_l, x_k)
def K(f_arr, ksi, x_arr, alpha):
    h = (b - a) / m
    fl = complex(0, 0)
    for x_i, f_i in zip(x_arr, f_arr):
        fl += x_i ** (alpha * ksi - 1) * f_i * h
    return fl


# Входной сигнал f(x_k)
def f(x, beta):
    return np.exp(complex(0, 1) * beta * x)


# Основная функция для расчета F(ξ_l)
def calculate_F(f_arr, x_arr, ksi_arr, alpha):
    g_arr = np.zeros(m, dtype=complex)
    for i, ksi in enumerate(ksi_arr):
        g_arr[i] = K(f_arr, ksi, x_arr, alpha)
    return g_arr


def plot_graphs(x_arr, y_arr1, y_arr2, title1, title2, suptitle):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.plot(x_arr, y_arr1, color='b')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.plot(x_arr, y_arr2, color='b')
    plt.grid(True)

    plt.suptitle(suptitle, color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def generate_suptitle(param_name, param_value):
    return f'Current {param_name} = {param_value}'


def experiment_change_beta(x_arr):
    beta_values = [1/10, -10000, -10, -1, 1 / 100000, 1, 10, 10000]
    for beta_value in beta_values:
        y_abs = np.abs(f(x_arr, beta_value))
        y_angle = np.angle(f(x_arr, beta_value))

        plot_graphs(x_arr, y_abs, y_angle, Title.inputA.value, Title.inputF.value, generate_suptitle('β', beta_value))


def experiment_change_integration_area(x_arr, alpha):
    p_new = [-500, -100, -100, -10, 0, 100]
    q_new = [-100, 0, 100, 10, 100, 500]
    f_x = f(x_arr, beta)
    for p, q in zip(p_new, q_new):
        ksi_arr_new = np.linspace(p, q, m)
        g_array = calculate_F(f_x, x_arr, ksi_arr_new, alpha)
        plot_graphs(ksi_arr_new, np.abs(g_array), np.angle(g_array), Title.outputA.value, Title.outputF.value, generate_suptitle('[p, q]', f'[{p}, {q}]'))


# Эксперимент с изменением alpha
def experiment_change_alpha(x_arr, ksi_arr):
    alpha_values = [1 / 10000, 0.5, 1, 10, 100]
    f_x = f(x_arr, beta)
    for alpha_value in alpha_values:
        g_array = calculate_F(f_x, x_arr, ksi_arr, alpha_value)
        plot_graphs(ksi_arr, np.abs(g_array), np.angle(g_array), Title.outputA.value, Title.outputF.value, generate_suptitle('α', alpha_value))


def experiment_change_b(ksi_arr, alpha):
    b_values = [1 / 10000, 0.5, 1, 10, 100]
    for b_value in b_values:
        x_arr_new = np.linspace(a, b_value, n)
        g_array = calculate_F(f(x_arr_new, beta), x_arr_new, ksi_arr, alpha)
        plot_graphs(ksi_arr, np.abs(g_array), np.angle(g_array), Title.outputA.value, Title.outputF.value, generate_suptitle('[a, b]', f'[{a}, {b_value}]'))


if __name__ == '__main__':
    m, n = 1000, 1000
    a, b = 1, 5
    p, q = 0, 3
    alpha, beta = 1, 1 / 10

    x_arr = np.linspace(a, b, n)
    ksi_arr = np.linspace(p, q, m)

    # experiment_change_beta(x_arr)

    # g_array = calculate_F(f(x_arr, beta), x_arr, ksi_arr, alpha)
    # plot_graphs(ksi_arr, np.abs(g_array), np.angle(g_array), Title.outputA.value, Title.outputF.value, generate_suptitle('[p, q]', f'[{p}, {q}]'))

    # experiment_change_integration_area(x_arr, alpha)

    # experiment_change_alpha(x_arr, ksi_arr)

    # experiment_change_b(ksi_arr, alpha)
