import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, diff, lambdify


""" Задаем тестовую символьную функцию """
x_sym, y_sym = symbols('x y')
#f_sym = 0.26 * (x_sym ** 2 + y_sym ** 2) - 0.48 * x_sym * y_sym
f_sym = (x_sym + 2 * y_sym - 7) ** 2 + (2 * x_sym + y_sym - 5) ** 2


# Вычисление градиента с помощью sympy
grad_x_sym = diff(f_sym, x_sym)  # Частная производная по x
grad_y_sym = diff(f_sym, y_sym)  # по y


# Преобразование символьных выражений в функции
grad_x_func = lambdify((x_sym, y_sym), grad_x_sym)
grad_y_func = lambdify((x_sym, y_sym), grad_y_sym)


# Преобразование символьной функции в численную
f = lambdify((x_sym, y_sym), f_sym)


""" Реализация градиентного спуска с моментной модификацией и методом затухания lr в процессе инкрементации """
def gradient_descent_evolved(start_x, start_y, init_learning_rate=0.003, mom=0.7, iter=2000):
    x, y = start_x, start_y
    x_prev, y_prev = x, y
    history = [(x, y, f(x, y))]

    for i in range(iter):
        learning_rate = init_learning_rate / (1 + 0.1 * i)

        # Вычисление градиента
        df_dx = grad_x_func(x, y)
        df_dy = grad_y_func(x, y)

        x_temp, y_temp = x, y  # Сохраняем текущие значения x и y перед обновлением

        x = x - learning_rate * df_dx + mom * (x - x_prev)
        y = y - learning_rate * df_dy + mom * (y - y_prev)

        x_prev, y_prev = x_temp, y_temp

        # Проверка переполнения
        try:
            f_val = f(x, y)
            history.append((x, y, f_val))
        except OverflowError:
            print(f"Переполнение на итерации {i}: x={x}, y={y}")
            break

    return x, y, history


# Параметры начальной точки и алгоритма
start_x, start_y = 5, 5  # != аналитический оптимум
init_learning_rate = 0.003
mom = 0.7
iters = 2000


# Запуск градиентного спуска
x_opt, y_opt, history = gradient_descent_evolved(start_x, start_y, init_learning_rate, mom, iters)


# Вычисление погрешности
inaccuracy = np.sqrt((x_opt - 1) ** 2 + (y_opt - 3) ** 2) # поменять значения, если точка оптимума другая


print(f"Точка оптимума: x = {x_opt:.5f}, y = {y_opt:.5f}")
print(f"Минимум функции: f_min(x, y) = {f(x_opt, y_opt):.5f}")
print(f"Погрешность относительно аналитической точки оптимума: {inaccuracy:.5f}")


# Извлекаем все необходимые значения из массива
x_vals = [point[0] for point in history]
y_vals = [point[1] for point in history]
func_vals = [point[2] for point in history]


# Построение графика зависимости значения функции при приближении к точке оптимума
fig, ax = plt.subplots()
ax.plot(range(len(func_vals)), func_vals, label='f(x, y) --> min')
ax.set_xlabel('Итерация')
ax.set_ylabel('f(x, y)')
ax.set_title('Сходимость градиентного спуска')
plt.grid(True)
plt.legend()
plt.show()


# Визуализация траектории на плоскости (x, y)
#plt.figure(figsize=(-10, 10))
plt.plot(x_vals, y_vals, 'o-', label='Траектория', markersize=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектория градиентного спуска')
plt.grid(True)
plt.legend()
plt.show()


# Отображение графика тестовой функции и точки оптимума с анимацией
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
func_mesh = f(x_mesh, y_mesh)


fig = plt.figure(figsize=(10, 9)) # можно увеличить атрибуты для более наглядного приближения
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, func_mesh, cmap='viridis', alpha=0.7)


# Добавление анимации для отображения траектории
trajectory, = ax.plot([], [], [], 'r-', label='Траектория')
point, = ax.plot([], [], [], 'bo', markersize=8, label='Текущая точка')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('График тестовой функции и траектория градиентного спуска')
plt.legend()


# Функция для анимации движения точки по траектории
def update(num):
    trajectory.set_data(x_vals[:num], y_vals[:num])
    trajectory.set_3d_properties(func_vals[:num])
    point.set_data([x_vals[num - 1]], [y_vals[num - 1]])
    point.set_3d_properties([func_vals[num - 1]])


ani = FuncAnimation(fig, update, frames=len(x_vals), interval=100, repeat=False)
plt.show()
