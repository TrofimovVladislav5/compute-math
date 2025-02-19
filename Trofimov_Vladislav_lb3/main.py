import math
import numpy as np
from scipy.optimize import bisect as scipy_bisect
import matplotlib.pyplot as plt


# Вариант 20: функция f(x) = exp(-x) - x^3
def f(x):
    return math.exp(-x) - x ** 3


def Round(x, delta):
    if x == 0.0 or delta == 0.0:
        return x  # без округления
    if x > 0.0:
        return delta * int((x / delta) + 0.5)
    else:
        return delta * int((x / delta) - 0.5)

def Bisect(left, right, epsilon, delta):
    iter_count = 0
    while (right - left) / 2 > epsilon:
        middle = (left + right) / 2
        f_middle = Round(f(middle), delta)
        f_left = Round(f(left), delta)

        if f_middle == 0:
            return middle, iter_count

        if f_left * f_middle < 0:
            right = middle
        else:
            left = middle

        iter_count += 1

    return (left + right) / 2, iter_count


def explore_iterations_by_eps_and_delta(left, right, eps_values, delta_values):
    iterations_by_delta = {}

    for delta in delta_values:
        iterations = []
        for eps in eps_values:
            _, iter_count = Bisect(left, right, eps, delta)
            iterations.append(iter_count)
        iterations_by_delta[delta] = iterations

    return iterations_by_delta


def plot_iterations(left, right, eps_values, delta_values):
    iterations_by_delta = explore_iterations_by_eps_and_delta(left, right, eps_values, delta_values)
    plt.figure(figsize=(10, 6))
    for delta, iterations in iterations_by_delta.items():
        plt.plot(eps_values, iterations, marker='o', label=f'Delta = {delta}')

    plt.xscale('log')  # Логарифмическая шкала по оси X
    plt.xlabel('Eps (точность)')
    plt.ylabel('Число итераций')
    plt.title('Зависимость числа итераций от точности Eps')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_bisect(left, right):
        eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        for eps in eps_values:
            root_custom, _ = Bisect(left, right, eps, delta=0)
            root_scipy = scipy_bisect(f, left, right, xtol=eps)
            assert abs(root_custom - root_scipy) < eps, f"Ошибка: {root_custom} != {root_scipy} при eps={eps}"
            print(f"eps={eps}: {root_custom} == {root_scipy}")

        prev_iters = 0
        for eps in eps_values:
            _, iter_count = Bisect(left, right, eps, delta=0)
            assert iter_count >= prev_iters, f"Ошибка: итерации уменьшились при eps={eps}"
            prev_iters = iter_count

        print("Тест 1: Проверка корня (должен совпадать с scipy.optimize.bisect)")
        print("Тест 2: Проверка числа итераций (должно увеличиваться при уменьшении eps)")

def main():
    left, right = map(float,input().split())  # Начальные границы отрезка
    test_bisect(left, right)
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  # Множество значений Eps
    delta_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  # Множество значений Delta для моделирования ошибок
    plot_iterations(left, right, eps_values, delta_values)

main()