from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def gauss_beam(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Гауссов пучок"""
    return np.exp(-x ** 2)


def plot_signal(x: NDArray[np.float64], signal: NDArray[np.complex128], title: str, phase: bool = False) -> None:
    """График сигнала"""
    plt.figure()
    data = np.angle(signal) if phase else np.abs(signal)
    plt.plot(x, data)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fft_process(signal: NDArray[np.float64], M: int, N: int, hx: float) -> NDArray[np.complex128]:
    """Обработка сигнала через БПФ"""
    padded_signal = np.pad(signal, pad_width=(M - N) // 2, mode='constant')
    FFT_signal = np.fft.fftshift(np.fft.fft(np.fft.fftshift(padded_signal))) * hx
    return FFT_signal[M // 2 - N // 2:M // 2 + N // 2]


def fourier_integral(x: NDArray[np.float64], u: NDArray[np.float64], signal: NDArray[np.float64], hx: float) -> NDArray[
    np.complex128]:
    """Преобразование Фурье через интеграл"""
    X, U = np.meshgrid(x, u)
    Kernel = np.exp(-2j * np.pi * X * U)
    return Kernel @ signal * hx


def input_field(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Входное поле rect((x + 2) / 2)"""
    return np.where(np.abs(x + 2) < 1, 1, 0)


def analytical_solution(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Аналитическое решение"""
    return 4 * np.exp(2 * 1j * np.pi * u) * np.sinc(4 * u)


def fft_2d_process(field: NDArray[np.float64], M: int, N: int, hx: float) -> NDArray[np.complex128]:
    """Обработка двумерного поля через БПФ"""
    field = np.apply_along_axis(fft_process, axis=0, arr=field, M=M, N=N, hx=hx)
    field = np.apply_along_axis(fft_process, axis=1, arr=field, M=M, N=N, hx=hx)
    return field


def plot_2d_results(field: NDArray[np.complex128], title: str, extent: Optional[List[float]] = None) -> None:
    """Построение двумерных графиков амплитуды и фазы"""
    extent = extent if extent else (-5, 5, -5, 5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(np.abs(field), extent=extent, cmap='jet', aspect='auto')
    axes[0].set_title(f'Амплитуда {title}')
    axes[0].figure.colorbar(axes[0].images[0], ax=axes[0])

    axes[1].imshow(np.angle(field), extent=extent, cmap='jet', aspect='auto')
    axes[1].set_title(f'Фаза {title}')
    axes[1].figure.colorbar(axes[1].images[0], ax=axes[1])

    fig.suptitle(f'Амплитуда и фаза {title}', color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_results(x: np.ndarray, signal: np.ndarray, title: str) -> None:
    """Построение графиков амплитуды и фазы"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(x, np.abs(signal), color='b')
    axes[0].set_title(f'Амплитуда {title}')
    axes[0].grid(True)

    axes[1].plot(x, np.angle(signal), color='b')
    axes[1].set_title(f'Фаза {title}')
    axes[1].grid(True)

    fig.suptitle(f'Амплитуда и фаза {title}', color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_results_multiple(x: np.ndarray, signals: List[NDArray[np.complex128]], titles: List[str]) -> None:
    """Построение графиков амплитуды и фазы для нескольких сигналов"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for signal, label in zip(signals, titles):
        axes[0].plot(x, np.abs(signal), label=label)
        axes[1].plot(x, np.angle(signal), label=label)

    axes[0].set_title('Амплитуда')
    axes[1].set_title('Фаза')

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.suptitle('Сравнение амплитуд и фаз', color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def experiment_change_N_M(signal: NDArray[np.float64], a: float) -> None:
    """
    Исследование влияния параметров N и M на результат БПФ.

    Parameters:
    -----------
    signal : np.ndarray
        Входной сигнал (например, гауссов пучок).
    a : float
        Граница области определения сигнала.
    """
    # Значения N и M для исследования
    N_values = np.array([128, 256, 512])  # Примеры значений N
    M_values = np.array([128, 256, 512])  # Примеры значений M

    for N in N_values:
        hx = (2 * a) / N
        x = np.linspace(-a, a - hx / 2, N)

        for M in M_values:
            hu = (2 * (N ** 2 / (4 * a * M))) / N
            u = np.linspace(-M * hu / 2, M * hu / 2 - hu, M)

            # Применяем БПФ
            FFT_signal = fft_process(signal, M, N, hx)

            # Строим графики амплитуды и фазы
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.title(f'Амплитуда N={N}, M={M}')
            plt.plot(u, np.abs(FFT_signal), color='b')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.title(f'Фаза N={N}, M={M}')
            plt.plot(u, np.angle(FFT_signal), color='b')
            plt.grid(True)

            plt.suptitle(f'Исследование влияния N и M на БПФ (N={N}, M={M})', color='r')
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            plt.show()


def main() -> None:
    # Входные параметры
    M, N = 1024, 200
    a = 5
    b = (N ** 2) / (4 * a * M)
    hx = (2 * a) / N
    hu = (2 * b) / N
    x = np.linspace(-a, a - hx / 2, N)
    u = np.linspace(-b, b - hu / 2, N)

    # Гауссов пучок
    gauss = gauss_beam(x)
    plot_results(x, gauss, 'Гауссов пучок')

    FFT_gauss = fft_process(gauss, M, N, hx)
    plot_results(u, FFT_gauss, 'БПФ Гауссова пучка')

    G = fourier_integral(x, u, gauss, hx)
    plot_results(u, G, 'ПФ Гауссова пучка')

    # Входное поле
    f = input_field(x)
    plot_results(x, f, 'Входное поле')

    # БПФ входного поля
    FFT2 = fft_process(f, M, N, hx)
    plot_results(u, FFT2, 'БПФ входного поля')

    # Входное поле - аналитическое решение
    FA = analytical_solution(u)
    plot_results_multiple(u, [FA, FFT2], ['Аналитическое решение', 'БПФ входного поля'])

    # Двумерный гауссов пучок
    X, Y = np.meshgrid(x, x)
    gauss2 = np.exp(-(X ** 2 + Y ** 2))
    plot_2d_results(gauss2, 'Двумерный Гауссов пучок')

    # БПФ двумерного гауссова пучка
    gauss2 = fft_2d_process(gauss2, M, N, hx)
    plot_2d_results(gauss2, 'БПФ двумерного Гауссова пучка')

    # Двумерное входное поле
    f2 = np.zeros_like(X)
    f2[(np.abs(X + 2) < 1) & (np.abs(Y + 2) < 1)] = 1
    plot_2d_results(f2, 'Двумерное входное поле')

    # БПФ двумерного входного поля
    f2 = fft_2d_process(f2, M, N, hx)
    plot_2d_results(f2, 'БПФ двумерного входного поля')

    # Аналитическое двумерное решение
    FA2 = np.outer(FA, FA)
    plot_2d_results(FA2, 'Аналитическое двумерное решение')

    experiment_change_N_M(gauss, a)


if __name__ == '__main__':
    main()
