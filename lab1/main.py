import math
import matplotlib.pyplot as plt
import itertools
import types
from typing import List
from prettytable import PrettyTable

import os

# Consts
PI = math.pi
piLog2_5PI = PI * math.log2(5 * PI)

hoard_table = PrettyTable()
newton_table = PrettyTable()
hoard_table.field_names = ["Iteration", "First interval", "Second interval", "X value"]
newton_table.field_names = ["Iteration", "hk", "Xk-1", "Xk"]
'''
    Zinchuk Mykhailo KV-94 Var 7 ->  22
    Hoard method
    Discrete method Newton

    f1 -> x^2 + pi*log2(5*pi) - 7*pi*cos(x) - 3*x = 0

    f2 -> e^x + sin(x) - 10x + x^10 + e^cos(x) = 0
   
'''

f1 = types.SimpleNamespace(
    as_str="x^2 + pi*log2(5*pi) - 7*pi*cos(x) - 3*x",
    as_func=lambda x: x ** 2 + piLog2_5PI - 7 * PI * math.cos(x) - 3 * x,
    as_derivative1_str="2x + 7pi*sin(x) -3",
    as_derivative1=lambda x: 2 * x + 7 * PI * math.sin(x) - 3,
    as_derivative2_str="2 + 7 * M_PI * cos(x)",
    as_derivative2=lambda x: 2 + 7 * PI * math.cos(x),
    boundaries=[],
)
f2 = types.SimpleNamespace(
    as_str="e^x + sin(x) - 10x + x^10 + e^cos(x) = 0",
    as_func=lambda x: math.exp(x) + math.sin(x) - 10 * x + x ** 10 + math.exp(math.cos(x)),
    as_derivative1_str="e^x + сos(x) - 10 + 10*x^9 - e^(cos(x))*sin(x)",
    as_derivative1=lambda x: math.exp(x) + math.cos(x) - 10 + 10 * x ** 9 - math.exp(math.cos(x)) * math.sin(x),
    as_derivative2_str="e^x - sin(x) + 90x^8 + e^(cos(x))  * sin(x)^2 - e^cos(x) * cos(x)",
    as_derivative2=lambda x: math.exp(x) - math.sin(x) + 90 * x ** 8 + math.exp(math.cos(x)) * (
        math.sin(x)) ** 2 - math.exp(
        math.cos(x)) * math.cos(x),
    boundaries=[],
)


def draw_graph(x1, x2, f, title, f_name):
    """
    Draw a function x^2 + pi*log2(5*pi) - 7*pi*cos(x) - 3*x = 0
    :param f_name:  name for file
    :param f: function
    :param title: title
    :param x1: left x for graphic
    :param x2: right x for graphic
    """
    x_arr = []
    y_arr = []
    for x in itertools.count(x1, 0.00001):
        x_arr.append(x)
        y = f(x)
        y_arr.append(y)
        if x >= x2:
            break
    plt.title(title)
    plt.plot(x_arr, y_arr)
    plt.grid(True)
    plt.savefig(f"./lab1/results/{f_name}")
    plt.show()


# drawing graphic
draw_graph(-1, 4, f1.as_func, f1.as_str, "f1_graphic.png")
draw_graph(-0.5, 1.3, f2.as_func, f2.as_str, "f2_graphic.png")


def get_intervals(f, rec1, rec2):
    print(f"\nEnter the  boundaries for {f.as_str}\n "
          f"Recommended is {str(rec1)} {str(rec2)}\n")
    a = input("[a, b], [a, b] -> enter with a space between -> for example(0 1 2 4) ")
    bound = []
    bound.extend(map(float, a.split(" ")))
    return bound


# isolation interval found from graphic
bnd = get_intervals(f1, [-2.0, 0], [1.0, 1.5] )
f1.boundaries.append(bnd[:2])
f1.boundaries.append(bnd[2:])

bnd = get_intervals(f2, [0.5, 0.75], [1.0, 1.25])
f2.boundaries.append(bnd[:2])
f2.boundaries.append(bnd[2:])


def hoard_method(a: float, b: float, eps: float, f) -> List[float]:
    """
    Find a value of root with khord method
    :param f: function method
    :param a: -> [a, b] interval value
    :param b:  -> [a, b] interval value
    :param eps: -> accuracy
    :return: result of operations
    """
    result = list()

    for i in itertools.count(0):
        c = a - (f(a) * (b - a)) / (f(b) - f(a))
        result.append(c)
        hoard_table.add_row([i, a, b, c])
        if len(result) != 1:
            if math.fabs(c - result[-2]) < eps:
                return result

        if f(a) * f(c) < 0:
            b = c
        elif f(c) * f(b) < 0:
            a = c
        else:
            raise ValueError


def check(a: float, b: float, f, d2f) -> float:
    """
    Function finding initial approximating(Fourier condition)
    :param f:
    :param d2f:
    :param a: left part of interval
    :param b: right part of interval
    :return: a or b
    """
    if (f(a) * d2f(a)) > 0:
        return a

    if (f(b) * d2f(b)) > 0:
        return b


def get_k(i):
    return 1 / (2 ** i)


def newton_method(a: float, b: float, eps: float, f) -> List[float]:
    """
    Realization of Newton method
    :param d2f: derivative
    :param f:  function
    :param a: -> [a, b]
    :param b: -> [a, b]
    :param eps: accuracy
    :return: List of results on every iteration
    """
    result = list()
    h = 0
    val = check(a, b, f2.as_func, f2.as_derivative2)
    result.append(val)
    xk_1 = None
    i = 0
    while True:
        if xk_1 is not None:
            xk = xk_1
        else:
            xk = val
        h = get_k(i)
        i += 1
        xk_1 = xk - ((f(xk) * h) / (f(xk + h) - f(xk)))
        newton_table.add_row([i, h, xk_1, xk])
        result.append(xk_1)
        if math.fabs(xk_1 - xk) < eps:
            break
    return result


def draw_results(result1: List[float], result2: List[float], title, f_nms: types.SimpleNamespace, f_name):
    """
    Drawing plot of iterations and roots
    :param result1: result list for first root
    :param result2: result list for second root
    :param title:  title for plot
    :param f_nms:  name for file
    """
    plt.figure()
    plt.suptitle(title)
    iterations1 = list()
    iterations2 = list()
    for i in range(1, len(result1) + 1):
        iterations1.append(i)

    for i in range(1, len(result2) + 1):
        iterations2.append(i)

    plt.subplot(223)
    plt.plot(iterations1, result1, 'bo', iterations1, result1, 'b')
    plt.xlabel('iteration')
    plt.ylabel('root')
    plt.title("x1 Є " + str(f_nms.boundaries[0]) + "\nX1 = " + str(result1[-1]))
    plt.grid(True)

    plt.subplot(224)
    plt.plot(iterations2, result2, 'bo', iterations2, result2, 'b')
    plt.xlabel('iteration')
    plt.ylabel('root')
    plt.title("x2 Є " + str(f_nms.boundaries[1]) + "\nX2 = " + str(result2[-1]))
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.savefig(f"./lab1/results/{f_name}")
    plt.show()


def print_info(title, table, results):
    print(f"___________________{title}__________________")
    print(table)
    table.clear_rows()
    print("Result: " + str(results[-1]) + "\n")


hoard_result_1_interval = hoard_method(f1.boundaries[0][0], f1.boundaries[0][1], 10 ** -7, f1.as_func)
print_info("Hoard method", hoard_table, hoard_result_1_interval)

hoard_result_2_interval = hoard_method(f1.boundaries[1][0], f1.boundaries[1][1], 10 ** -7, f1.as_func)
print_info("Hoard method", hoard_table, hoard_result_2_interval)

newton_result_1_interval = newton_method(f2.boundaries[0][0], f2.boundaries[0][1], 10 ** -7, f2.as_func)
print_info("Discrete Newton method", newton_table, newton_result_1_interval)

newton_result_2_interval = newton_method(f2.boundaries[1][0], f2.boundaries[1][1], 10 ** -7, f2.as_func)
print_info("Discrete Newton method", newton_table, newton_result_2_interval)

draw_results(hoard_result_1_interval, hoard_result_2_interval,
             "Hoard method \n" + f1.as_str, f1, "hoard_method.png")

draw_results(newton_result_1_interval, newton_result_2_interval,
             "Discrete Newton method \n" + f2.as_str, f2, "newton_method.png")

# print("""
#                     Hoard method
#        x^2 + pi*log2(5*pi) - 7*pi*cos(x) - 3*x = 0\n"""
#       )
#
# print(f"x1 ->  {hoard_method(f1.boundaries[0][0], f1.boundaries[0][1], 10 ** -7, f1.as_func)[-1]} - {f1.boundaries[0]}")
# print(f"x2 ->  {hoard_method(f1.boundaries[1][0], f1.boundaries[1][1], 10 ** -7, f1.as_func)[-1]} - {f1.boundaries[1]}")
#
# print("""\n
#                 Discrete Newton method
#         e^x + sin(x) - 10x + x^10 + e^cos(x) = 0
# """)
#
# print(
#     f"\n x1 -> {newton_method(f2.boundaries[0][0], f2.boundaries[0][1], 10 ** -7, f2.as_func)[-1]} - {f2.boundaries[0]}")
# print(
#     f" x2 -> {newton_method(f2.boundaries[1][0], f2.boundaries[1][1], 10 ** -7, f2.as_func)[-1]} - {f2.boundaries[1]}")
