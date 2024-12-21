import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


""" Пользователь определяет тестовую функцию, для ручной подачи функции градиента выберем функцию Матьяса """
def f(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def input_gradient():
    print("Введите частные производные функции по переменным x и y:")
    df_dx_input = input("Частная производная по x (для заданной тестовой функции): ") # 0.52 * x - 0.48 * y
    df_dy_input = input("Частная производная по y (для заданной тестовой функции): ") # 0.52 * y - 0.48 * x

    def grad_x(x, y):
        return eval(df_dx_input)

    def grad_y(x, y):
        return eval(df_dy_input)

    return grad_x, grad_y

grad_x, grad_y = input_gradient()


def gradient_descent(start_x, start_y, learning_rate=0.1, iter=2500):
    x, y = start_x, start_y
    history = [(x, y, f(x, y))]

    for i in range(iter):

        df_dx = grad_x(x, y)
        df_dy = grad_y(x, y)

        x = x - learning_rate * df_dx
        y = y - learning_rate * df_dy

        history.append((x, y, f(x, y)))

    return x, y, history


# Параметры начальной точки и алгоритма
start_x, start_y = 1, 8  # != аналитический оптимум
learning_rate = 0.1
iters = 2500


# Запуск градиентного спуска
x_opt, y_opt, history = gradient_descent(start_x, start_y, learning_rate, iters)


# Вычисление погрешности
inaccuracy = np.sqrt((x_opt - 0) ** 2 + (y_opt - 0) ** 2)


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
