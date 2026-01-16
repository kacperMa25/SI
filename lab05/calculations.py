# pyright: standard
import math
import matplotlib.pyplot as plt
import numpy as np

w1 = 0.5
w2 = -0.5
b = 0
T = 100
n = 0.01

x1 = 1
x2 = 2
y = 1


def sigmoid(z) -> float:
    return 1 / (1 + math.e ** (-z))


def loss_function(y, y_pred) -> float:
    return -(y * math.log(y_pred) + (1 - y) * math.log(1 - y_pred))


print("Iteration 0:")
print(f"w1 = {w1}, w2 = {w2}, b = {b}")

params_sgd = {"w1": w1, "w2": w2, "b": b}
params_adam = {"w1": w1, "w2": w2, "b": b}

print("\n============SGD============")

for t in range(1, T + 1):
    z = params_sgd["w1"] * x1 + params_sgd["w2"] * x2 + params_sgd["b"]
    y_pred = sigmoid(z)
    loss = loss_function(y, y_pred)
    error = y_pred - y

    gradients = {
        "w1": error * x1,
        "w2": error * x2,
        "b": error,
    }

    for key in params_sgd:
        params_sgd[key] -= n * gradients[key]

    print(f"Iteration {t}:")
    print(f"Loss = {loss}")
    print(f"w1 = {params_sgd['w1']}, w2 = {params_sgd['w2']}, b = {params_sgd['b']}")


print("\n============Adam============")

B1 = 0.9
B2 = 0.999
epsilon = 10 ** (-8)

m = {key: 0 for key in params_adam}
v = {key: 0 for key in params_adam}

for t in range(1, T + 1):
    z = params_adam["w1"] * x1 + params_adam["w2"] * x2 + params_adam["b"]
    y_pred = sigmoid(z)
    loss = loss_function(y, y_pred)
    error = y_pred - y

    gradients = {
        "w1": error * x1,
        "w2": error * x2,
        "b": error,
    }

    for key in params_adam:
        m[key] = B1 * m[key] + (1 - B1) * gradients[key]
        v[key] = B2 * v[key] + (1 - B2) * (gradients[key] ** 2)

        m_hat = m[key] / (1 - B1**t)
        v_hat = v[key] / (1 - B2**t)

        params_adam[key] -= n * m_hat / (math.sqrt(v_hat) + epsilon)

    print(f"Iteration {t}:")
    print(f"Loss = {loss}")
    print(f"w1 = {params_adam['w1']}, w2 = {params_adam['w2']}, b = {params_adam['b']}")

plt.figure(figsize=(8, 6))
plt.title("Granica decyzyjna")
plt.xlabel("x1")
plt.ylabel("x2")

for params, name in zip((params_adam, params_sgd), ("Adam", "SGD")):
    w1 = params["w1"]
    w2 = params["w2"]
    b = params["b"]

    # Przestrzeń wartości x1
    x1_space = np.linspace(-1, 1, 100)

    x2_boundary = -(w1 / w2) * x1_space - (b / w2)

    plt.plot(x1_space, x2_boundary, label=name)

plt.scatter(x1, x2)
plt.legend()
plt.grid()
plt.show()
