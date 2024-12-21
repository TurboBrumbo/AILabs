import pygmo as pg
import numpy as np


""" Задание двух тестовых функций, они могут быть любыми, выбираются пользователем """
# В качестве первой тестовой функции возьмем функцию Матьяса
def test_func_1(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

# В качестве второй - функцию Розенброка
def test_func_2(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))


# Класс задачи для оптимизации
class Problem:
    def __init__(self, func, dim, bounds):
        self.func = func # Для выбора тестовой функции
        self.dim = dim # Для установки размерности функции
        self.bounds = bounds # Для установки допустимых границ значений функции

    def fitness(self, x):
        return [self.func(x)]

    def get_bounds(self):
        return self.bounds


# Настройка задачи для оптимизации
test_func_1_problem = pg.problem(Problem(test_func_1, 2, ([-10] * 2, [10] * 2)))
test_func_2_problem = pg.problem(Problem(test_func_2, 2, ([-2] * 2, [2] * 2)))


""" Выбор трех алгоритмов глобальной оптимизации, пользователь может выбирать любые из библиотеки """
algorithms = {
    "Differential Evolution": pg.algorithm(pg.de(gen=100)),
    "Particle Swarm Optimization": pg.algorithm(pg.pso(gen=100)),
    "Grey Wolf Optimizer": pg.algorithm(pg.gwo(gen=100))
} # Число итераций gen можно варьировать для достижения наилучшего результата


# Функция для выполнения оптимизации
def optimize(problem, algorithms):
    results = {}
    for name, algo in algorithms.items():
        algo.set_verbosity(False) # По желанию для подробного отображения процесса оптимизации можно поставить 1 или True
        pop = pg.population(problem, size=20)  # Создание начальной популяции
        # Размер size можно варьировать для достижения наилучшего результата
        pop = algo.evolve(pop)  # Выполнение оптимизации
        results[name] = (pop.champion_x, pop.champion_f[0])  # Лучшее решение для точки оптимума и значение функции
    return results


# Оптимизация для обеих задач
# print("_____ Оптимизация первой тестовой функции _____")
test_func_1_results = optimize(test_func_1_problem, algorithms)

# print("\n_____ Оптимизация второй тестовой функции _____")
test_func_2_results = optimize(test_func_2_problem, algorithms)


def print_results(problem_name, results):
    print(f"\nРезультаты для тестовой функции {problem_name}:")
    print(f"{'Алгоритм':<30} | {'Точка оптимума':<30} | {'Минимум функции':<15}")
    print("-" * 90)
    for algo, (x, f) in results.items():
        print(f"{algo:<30} | {np.array_str(np.array(x), precision=5):<30} | {f:<15.5f}")


# В problem_name вписать имя заданной тестовой функции
print_results("Матьяса", test_func_1_results)
print_results("Розенброка", test_func_2_results)
