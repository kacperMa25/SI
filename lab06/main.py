# pyright: standard
import random
import time
import numpy as np
from numpy.testing import print_assert_equal


class BoardVector:
    def __init__(self, size=4):
        self._init_board(size)

    def generate_board(self, size):
        self._init_board(size)

    def _init_board(self, size):
        self._size = size
        self._vector = [random.randint(0, size - 1) for _ in range(size)]

    def make_board(self, vector: list[int]):
        self._size = len(vector)
        self._vector = list(vector)

    def conflicts(self):
        counter = 0
        for i in range(self._size):
            for j in range(i + 1, self._size):
                if (
                    abs(self._vector[j] - self._vector[i]) == abs(j - i)
                    or self._vector[i] == self._vector[j]
                ):
                    counter += 1
        return counter

    def cross(self, other):
        idx = random.randint(1, self._size - 2)

        child1 = BoardVector(self._size)
        child2 = BoardVector(self._size)

        child1._vector = self._vector[:idx] + other._vector[idx:]
        child2._vector = other._vector[:idx] + self._vector[idx:]

        return child1, child2

    def mutate(self, pn):
        bv = BoardVector()
        return bv.make_board(
            [
                random.randint(0, self._size - 1) if random.random() < pn else gen
                for gen in self._vector
            ]
        )

    def measure_time(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflicts():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate


class Populatiuon:
    def __init__(self, pop=100, n=6):
        self._popSize = pop
        self._pop = []
        self._prevPop = []
        self._n = n

        for _ in range(self._popSize):
            bv = BoardVector(n)
            self._pop = [{"board": bv, "eval": 0}]

    def evaluate(self, pop):
        for indv in pop:
            indv["eval"] = indv["board"].conflicts()

    def crossover(self, pc):
        for i in range(0, self._popSize, 2):
            if random.random() < pc:
                child1, child2 = self._pop[i]["board"].cross(self._pop[i + 1]["board"])
                self._pop[i] = {"board": child1, "eval": 0}
                self._pop[i + 1] = {"board": child2, "eval": 0}

    def mutation(self, pn):
        for i in range(self._popSize):
            self._pop[i] = self._pop[i]["board"].mutate(pn)

    def selection(self):
        pass
