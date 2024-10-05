import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List


def gauss_beam(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Гауссов пучок"""
    return np.exp(-x ** 2)


def plot_signal(x: NDArray[np.float64], signal: NDArray[np.complex128], title: str, phase: bool = False) -> None:
    """График сигнала"""
    plt.figure()
    plt.plot(x, np.angle(signal) if phase else np.abs(signal))
    plt.title(title)


def fft_process(signal: NDArray[np.float64], M: int, N: int, hx: float) -> NDArray[np.complex128]:
    """Обработка сигнала через БПФ"""
    padded_signal = np.concatenate([np.zeros((M - N) // 2), signal, np.zeros((M - N) // 2)])
    FFT_signal = np.fft.fftshift(np.fft.fft(np.fft.fftshift(padded_signal))) * hx
    return FFT_signal[M // 2 - N // 2:M // 2 + N // 2]


def fourier_integral(x: NDArray[np.float64], u: NDArray[np.float64], signal: NDArray[np.float64], hx: float) -> NDArray[np.complex128]:
    """Преобразование Фурье через интеграл"""
    X, U = np.meshgrid(x, u)
    Kernel = np.exp(-2j * np.pi * X * U)
    return np.dot(Kernel, signal) * hx


def input_field(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Входное поле rect((x + 2) / 2)"""
    f = np.zeros_like(x)
    f[np.abs(x) < 2] = 1
    return f


def analytical_solution(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Аналитическое решение"""
    return 4 * np.sinc(4 * u)


def fft_2d_process(field: NDArray[np.float64], M: int, N: int, hx: float) -> NDArray[np.complex128]:
    """Обработка двумерного поля через БПФ"""
    for i in range(N):
        row_fft = fft_process(field[:, i], M, N, hx)
        field[:, i] = row_fft
    for i in range(N):
        col_fft = fft_process(field[i, :], M, N, hx)
        field[i, :] = col_fft
    return field


def plot_2d_results(field: NDArray[np.complex128], title: str, extent: List[float] = None) -> None:
    """Построение двумерных графиков амплитуды и фазы"""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Амплитуда {title}')
    plt.imshow(np.abs(field), extent=extent if extent else (-5, 5, -5, 5), cmap='jet', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f'Фаза {title}')
    plt.imshow(np.angle(field), extent=extent if extent else (-5, 5, -5, 5), cmap='jet', aspect='auto')
    plt.colorbar()

    plt.suptitle(f'Амплитуда и фаза {title}', color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_results(x: np.ndarray, signal: np.ndarray, title: str) -> None:
    """Построение графиков амплитуды и фазы"""
    plt.figure(figsize=(12, 6))

    # График амплитуды
    plt.subplot(1, 2, 1)
    plt.title(f'Амплитуда {title}')
    plt.plot(x, np.abs(signal), color='b')
    plt.grid(True)

    # График фазы
    plt.subplot(1, 2, 2)
    plt.title(f'Фаза {title}')
    plt.plot(x, np.angle(signal), color='b')
    plt.grid(True)

    plt.suptitle(f'Амплитуда и фаза {title}', color='r')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

def plot_results_multiple(x: np.ndarray, signals: List[NDArray[np.complex128]], titles: List[str]) -> None:
    """Построение графиков амплитуды и фазы для нескольких сигналов"""
    plt.figure(figsize=(12, 6))

    # График амплитуды
    plt.subplot(1, 2, 1)
    for signal, title in zip(signals, titles):
        plt.plot(x, np.abs(signal), label=title)
    plt.title('Амплитуда')
    plt.grid(True)
    plt.legend()

    # График фазы
    plt.subplot(1, 2, 2)
    for signal, title in zip(signals, titles):
        plt.plot(x, np.angle(signal), label=title)
    plt.title('Фаза')
    plt.grid(True)
    plt.legend()

    plt.suptitle('Сравнение амплитуд и фаз', color='r')
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
    plot_results(x, gauss, 'гауссова пучка')

    FFT_gauss = fft_process(gauss, M, N, hx)
    plot_results(u, FFT_gauss, 'БПФ гауссова пучка')

    G = fourier_integral(x, u, gauss, hx)
    plot_results(u, G, 'ПФ гауссова пучка')

    # Входное поле
    f = input_field(x)
    plot_results(x, f, 'входного поля')

    # БПФ входного поля
    FFT2 = fft_process(f, M, N, hx)
    plot_results(u, FFT2, 'БПФ входного поля')

    # Входное поле - аналитическое решение
    FA = analytical_solution(u)
    plot_results_multiple(u, [FA, FFT2], ['Аналитическое решение', 'БПФ входного поля'])

    # Двумерный гауссов пучок
    X, Y = np.meshgrid(x, x)
    gauss2 = np.exp(-(X ** 2 + Y ** 2))
    plot_2d_results(gauss2, 'Двумерный гауссов пучок')

    # БПФ двумерного гауссова пучка
    gauss2 = fft_2d_process(gauss2, M, N, hx)
    plot_2d_results(gauss2, 'БПФ двумерного гауссова пучка')

    # Двумерное входное поле
    f2 = np.zeros_like(X)
    f2[(np.abs(X) <= 2) & (np.abs(Y) <= 2)] = 1
    plot_2d_results(f2, 'Двумерное входное поле')

    # БПФ двумерного входного поля
    f2 = fft_2d_process(f2, M, N, hx)
    plot_2d_results(f2, 'БПФ двумерного входного поля')

    # Аналитическое двумерное решение
    FA2 = np.outer(FA, FA)
    plot_2d_results(FA2, 'Аналитическое двумерное решение')



if __name__ == '__main__':
    main()
