import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Число столбцов первой матрицы должно совпадать с числом строк второй матрицы.")

    result_rows = len(matrix_a)
    result_cols = len(matrix_b[0])

    result = [[0 for _ in range(result_cols)] for _ in range(result_rows)]

    for i in range(result_rows):
        for k in range(result_cols):
            for j in range(len(matrix_b)):
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """

    if a_1 == a_2:
        return None

    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    def F(x):
        return a11 * x ** 2 + a12 * x + a13

    if a11 != 0:
        x_extr_f = (-a12) / (2 * a11)
    if a21 != 0:
        x_extr_p = (-a22) / (2 * a21)

    solutions = []

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23

    discriminant = b ** 2 - 4 * a * c

    if a == 0:
        if b != 0:
            x = -c / b
            solutions.append((x, F(x)))
    elif discriminant > 0:
        x1 = (-b + discriminant ** 0.5) / (2 * a)
        x2 = (-b - discriminant ** 0.5) / (2 * a)
        solutions.append((x1, F(x1)))
        solutions.append((x2, F(x2)))

    return solutions


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x)

    m3 = np.sum((x - mean_x) ** 3) / n
    q = np.sum((x - mean_x) ** 2) / n

    A3 = m3 / (q ** (3 / 2))
    return round(A3, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x)

    m4 = np.sum((x - mean_x)**4) / n
    q = np.sum((x - mean_x) ** 2) / n

    E4 = m4 / (q ** 2) - 3
    return round(E4, 2)
