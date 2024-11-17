import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.special import factorial, jv
from tabulate import tabulate


def plot_data(data, title, x_values=None, extent=None, layout='vertical'):
    """Отображение амплитуды и фазы данных с использованием Matplotlib."""
    fig, axes = plt.subplots(2 if layout == 'vertical' else 1, 1 if layout == 'vertical' else 2,
                             figsize=(10, 6))
    axes = np.atleast_1d(axes)
    labels = ["Амплитуда", "Фаза"]

    for ax, component, label in zip(axes, [np.abs(data), np.angle(data)], labels):
        if extent is not None:
            img = ax.imshow(component, extent=extent)
            fig.colorbar(img, ax=ax)
        else:
            ax.plot(x_values, component)
        ax.set_title(label)
        ax.grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def fft_func(func, M, hx):
    """Выполнение быстрого преобразования Фурье (БПФ)."""
    N = len(func)
    padded_func = np.pad(func, (int((M - N) / 2), int((M - N) / 2)), 'constant')
    transformed_func = fftshift(fft(fftshift(padded_func))) * hx
    return transformed_func[int(M / 2 - N / 2): int(M / 2 + N / 2)]


def fft_2d_func(field, M, hx):
    """Обработка двумерного поля через БПФ"""
    field = np.apply_along_axis(fft_func, axis=0, arr=field, M=M, hx=hx)
    field = np.apply_along_axis(fft_func, axis=1, arr=field, M=M, hx=hx)
    return field


def radial_p(n, p, r):
    """Расчет радиальных полиномов Цернике."""
    R_np = 0
    for k in range(int((n - p) / 2) + 1):
        R_np += ((-1) ** k * factorial(n - k) /
                 (factorial(k) * factorial((n + p) // 2 - k) * factorial((n - p) // 2 - k))
                 ) * r ** (n - 2 * k)
    return R_np


def zernike_func(n, m, p, r):
    """Расчет полинома Цернике."""
    return radial_p(n, abs(p), r) * np.exp(1j * m)


def generate_image_from_zernike(zernike, N, m):
    """Восстановление двумерного изображения на основе полинома Цернике."""
    image = np.zeros((2 * N, 2 * N), dtype=complex)
    for row in range(2 * N):
        for col in range(2 * N):
            alpha = int(round(np.sqrt((row - N) ** 2 + (col - N) ** 2)))
            if alpha < N:
                image[row, col] = zernike[alpha] * np.exp(1j * m * np.arctan2(col - N, row - N))
    return image


def hankel_transform(zernike, r, hr, m):
    """Выполнение преобразования Ханкеля."""
    start_time = datetime.now()
    X, XI = np.meshgrid(r, r)
    A = (2 * np.pi / (1j ** m)) * jv(m, 2 * np.pi * X * XI) * X
    H = A.dot(zernike) * hr
    print(f'Время выполнения преобразования Ханкеля: {datetime.now() - start_time} сек')
    return H


def experiment(N_values, m):
    """Измерение времени выполнения для БПФ и преобразования Ханкеля с выводом в виде таблицы и графика."""
    results = []
    fft_times = []
    hankel_times = []

    for N in N_values:
        # Генерация случайного двумерного массива комплексных чисел
        image = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        r = np.linspace(0, 5, N)
        hr = 5 / N

        # Время выполнения двумерного БПФ
        start_time = time.perf_counter()
        fft_image = fft_2d_func(image, M=1024, hx=hr)
        fft_time = time.perf_counter() - start_time

        # Время выполнения преобразования Ханкеля
        zernike = zernike_func(5, m, -3, r)
        start_time = time.perf_counter()
        H = hankel_transform(zernike, r, hr, m)
        hankel_time = time.perf_counter() - start_time

        # Сохранение результатов для таблицы
        results.append([N, f"{fft_time:.4f}", f"{hankel_time:.4f}"])
        fft_times.append(fft_time)
        hankel_times.append(hankel_time)

    # Выводим результаты в виде таблицы
    print(tabulate(results, headers=["N", "БПФ время (сек)", "Ханкель время (сек)"], tablefmt="grid"))

    # Построение графика
    plot_results(N_values, fft_times, hankel_times)


def plot_results(N_values, fft_times, hankel_times):
    """Построение графика времени выполнения для БПФ и преобразования Ханкеля."""
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, fft_times, label="БПФ время (сек)", marker="o")
    plt.plot(N_values, hankel_times, label="Ханкель время (сек)", marker="o")
    plt.xlabel("Размерность N")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Сравнение времени выполнения БПФ и преобразования Ханкеля")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    m = -3  # Порядок полинома Цернике
    N = 128  # Число точек в радиусе
    R = 5  # Радиус
    hr = R / N  # Шаг по радиусу
    r = np.linspace(0, R - hr / 2, N)  # Радиусные точки

    # Расчет полинома Цернике и отображение амплитуды и фазы
    zernike = zernike_func(5, m, -3, r)
    plot_data(zernike, "Полином Цернике", x_values=r)

    # Восстановление изображения в двумерный массив
    image = generate_image_from_zernike(zernike, N, m)
    plot_data(image,
              "Амплитуда и фаза восстановленного изображения", extent=[-R, R, -R, R], layout='horizontal')

    # Преобразование Ханкеля и его отображение
    H = hankel_transform(zernike, r, hr, m)
    plot_data(H, "Преобразование Ханкеля", x_values=r)

    # Восстановление изображения из преобразования Ханкеля
    image_hankel = generate_image_from_zernike(H, N, m)
    plot_data(image_hankel,
              "Амплитуда и фаза после преобразования Ханкеля", extent=[-R, R, -R, R], layout='horizontal')

    # Двумерное преобразование Фурье через БПФ
    M = 1024
    b = N ** 2 / (4 * R * M)
    start_time = datetime.now()

    # Преобразование по строкам и столбцам
    for row in range(image.shape[0]):
        image[row, :] = fft_func(image[row, :], M, hr)
    for col in range(image.shape[1]):
        image[:, col] = fft_func(image[:, col], M, hr)

    print(f'Время выполнения преобразования Фурье: {datetime.now() - start_time} сек')

    # Отображение результатов преобразования Фурье
    plot_data(image,
              "Амплитуда и фаза после преобразования Фурье", extent=[-b, b, -b, b], layout='horizontal')

    N_values = [64, 128, 256, 512]
    experiment(N_values, m)


if __name__ == "__main__":
    main()
