# pyright: standard
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def loss_function(y, y_pred):
    return -(y * math.log(y_pred) + (1 - y) * math.log(1 - y_pred))


class Neuron:
    def __init__(self, w1, w2, b, n=0.01):
        self.params = {"w1": w1, "w2": w2, "b": b}
        self.n = n
        self.m = {key: 0 for key in self.params}
        self.v = {key: 0 for key in self.params}

    def compute_z(self, x1, x2):
        return self.params["w1"] * x1 + self.params["w2"] * x2 + self.params["b"]

    def update_weights_sgd(self, gradients):
        for key in self.params:
            self.params[key] -= self.n * gradients[key]

    def update_weights_adam(self, gradients, t, B1=0.9, B2=0.999, epsilon=10 ** (-8)):
        for key in self.params:
            self.m[key] = B1 * self.m[key] + (1 - B1) * gradients[key]
            self.v[key] = B2 * self.v[key] + (1 - B2) * (gradients[key] ** 2)

            m_hat = self.m[key] / (1 - B1**t)
            v_hat = self.v[key] / (1 - B2**t)

            self.params[key] -= self.n * m_hat / (math.sqrt(v_hat) + epsilon)

    def get_params(self):
        return self.params


class NeuralNetwork:
    def __init__(self):
        self.hidden_neuron1 = Neuron(0.2, -0.1, 0)
        self.hidden_neuron2 = Neuron(0.4, 0.3, 0)
        self.output_neuron = Neuron(0.5, -0.4, 0)

    def forward(self, x1, x2):
        z1 = self.hidden_neuron1.compute_z(x1, x2)
        a1 = sigmoid(z1)

        z2 = self.hidden_neuron2.compute_z(x1, x2)
        a2 = sigmoid(z2)

        z3 = self.output_neuron.compute_z(a1, a2)
        a3 = sigmoid(z3)

        return z1, a1, z2, a2, z3, a3

    def train(self, x1, x2, y, iterations=1, method="Adam"):
        for t in range(1, iterations + 1):
            z1, h1, z2, h2, z3, y_hat = self.forward(x1, x2)
            loss = loss_function(y, y_hat)
            error = sigmoid(z3) - y

            gradients_output = {
                "w1": error * h1,
                "w2": error * h2,
                "b": error,
            }

            error_hidden1 = (
                error * self.output_neuron.params["w1"] * sigmoid_derivative(z1)
            )
            error_hidden2 = (
                error * self.output_neuron.params["w2"] * sigmoid_derivative(z2)
            )

            gradients_hidden1 = {
                "w1": error_hidden1 * x1,
                "w2": error_hidden1 * x2,
                "b": error_hidden1,
            }

            gradients_hidden2 = {
                "w1": error_hidden2 * x1,
                "w2": error_hidden2 * x2,
                "b": error_hidden2,
            }

            if method == "SGD":
                self.output_neuron.update_weights_sgd(gradients_output)
                self.hidden_neuron1.update_weights_sgd(gradients_hidden1)
                self.hidden_neuron2.update_weights_sgd(gradients_hidden2)
            elif method == "Adam":
                self.output_neuron.update_weights_adam(gradients_output, t)
                self.hidden_neuron1.update_weights_adam(gradients_hidden1, t)
                self.hidden_neuron2.update_weights_adam(gradients_hidden2, t)
            else:
                print("Method isn't implemented")
                return

            print(f"\nEpoch {t}/{iterations}, Loss: {loss}, Output: {y_hat}")
            print("Hidden Neuron 1:", self.hidden_neuron1.get_params())
            print("Hidden Neuron 2:", self.hidden_neuron2.get_params())
            print("Output Neuron:", self.output_neuron.get_params())


def main():
    x1 = 1
    x2 = 2
    y = 1

    methods = ["SGD", "Adam"]
    iterations = 5
    for method in methods:
        print(f"\n{method}")
        network = NeuralNetwork()
        network.train(x1, x2, y, iterations=iterations, method=method)


main()
