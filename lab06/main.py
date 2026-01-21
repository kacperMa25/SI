# pyright: standard
from operator import ge
import random
import time
from typing import Dict, List, Set
from numpy import arange, ceil
import matplotlib.pyplot as plt
from pathlib import Path


class BoardVector:
    def __init__(self, size=4, seed=0):
        self._init_board(size, seed)

    def generate_board(self, size, seed=0):
        self._init_board(size, seed)

    def _init_board(self, size, seed):
        if seed != 0:
            random.seed(seed)

        self._size = size
        self._vector = [random.randint(0, size - 1) for _ in range(size)]

    def make_board(self, vector: list[int]):
        self._size = len(vector)
        self._vector = list(vector)

    def printBoard(self):
        for i in self._vector:
            print(i, end=" ")
        print()

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

    def cross(self, other, points: int = 2):
        cuts = sorted(random.sample(range(1, self._size), points))

        child1 = BoardVector(self._size)
        child2 = BoardVector(self._size)

        child1._vector = []
        child2._vector = []

        src1, src2 = self._vector, other._vector
        last = 0
        swap = False

        for cut in cuts + [self._size]:
            if not swap:
                child1._vector.extend(src1[last:cut])
                child2._vector.extend(src2[last:cut])
            else:
                child1._vector.extend(src2[last:cut])
                child2._vector.extend(src1[last:cut])
            swap = not swap
            last = cut

        return child1, child2

    def mutate(self, pn):
        bv = BoardVector()
        bv.make_board(
            [
                random.randint(0, self._size - 1) if random.random() < pn else gen
                for gen in self._vector
            ]
        )
        return bv

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
    def __init__(self, popSize=100, n=6, seed=0):
        self._pop: List[Dict] = []
        self._n = n

        for _ in range(popSize):
            bv = BoardVector(n, seed)
            self._pop.append({"board": bv, "eval": bv.conflicts()})

    def evaluate(self, pop):
        for i, indv in enumerate(pop):
            indv["eval"] = indv["board"].conflicts()

    def crossover(self, P: List[Dict], pc: float, points: int):
        Pn = []
        for i in range(0, len(P), 2):
            if random.random() < pc and i != len(P) - 1:
                child1, child2 = P[i]["board"].cross(P[i + 1]["board"], points)
                Pn.append({"board": child1, "eval": 0})
                Pn.append({"board": child2, "eval": 0})
            else:
                Pn.append(P[i])
                if i != len(P) - 1:
                    Pn.append(P[i + 1])
        return Pn

    def mutation(self, P: List[Dict], pn: float):
        Pn = []
        for i in range(len(P)):
            Pn.append({"board": P[i]["board"].mutate(pn), "eval": 0})
        return Pn

    def selection(self, P: List[Dict], indivPerRound: int = 3):
        Pn = []
        for i in range(len(P)):
            tournamentRound = [
                P[random.randint(0, len(P) - 1)] for _ in range(indivPerRound)
            ]
            _, best = self.bestCandidate(tournamentRound)
            Pn.append(best)
        return Pn

    def bestCandidate(self, P: List[Dict]) -> tuple[int, Dict]:
        return min(enumerate(P), key=lambda indiv: indiv[1]["eval"])

    def worstCandidate(self, P: List[Dict]) -> tuple[int, Dict]:
        return max(enumerate(P), key=lambda indiv: indiv[1]["eval"])

    def replacement(self, P: List[Dict], best: Dict):
        self._pop = P
        idx, _ = self.worstCandidate(self._pop)
        self._pop[idx] = best

    def evolve(self, pn=0.2, pc=0.8, genMax=10000, indivPerRound=3, points: int = 2):
        gen = 0
        _, best = self.bestCandidate(self._pop)
        meanFit = []
        bestFit = []

        while gen < genMax and best["eval"] > 0:
            meanFit.append(sum(indv["eval"] for indv in self._pop) / len(self._pop))
            bestFit.append(best["eval"])
            Pn = self.selection(self._pop, indivPerRound)
            Pn = self.crossover(Pn, pc, points)
            Pn = self.mutation(Pn, pn)
            self.evaluate(Pn)

            _, cand = self.bestCandidate(Pn)
            if cand["eval"] < best["eval"]:
                best = cand

            self.replacement(Pn, best)
            self.evaluate(self._pop)

            gen += 1

        meanFit.append(sum(indv["eval"] for indv in self._pop) / len(self._pop))
        bestFit.append(best["eval"])

        return best, gen, meanFit, bestFit

    def printPopulation(self):
        for indv in self._pop:
            print(f"{indv['eval']}: ", end="")
            indv["board"].printBoard()


def baseExperiment(howManyTimes: int):
    nHetman = [5, 10, 50, 100]
    path = "./csv/baseline_n.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for n in nHetman:
            seed = random.randint(1, 10000)
            for _ in range(howManyTimes):
                pop = Populatiuon(10, n=n, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(
                    points=int(ceil(n * 0.3)), genMax=1000
                )
                print(f"Best eval: {best['eval']}, Generation: {gen}, n: {n}")

                x = arange(gen + 1)

                for xi in x:
                    f.write(
                        f"{xi}, {seed}, {n}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


def popSizeExperiment(howManyTimes=5):
    popSizes = [10, 50, 100, 1000]
    path = "./csv/population.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for _ in range(howManyTimes):
            seed = random.randint(0, 10000)
            for size in popSizes:
                pop = Populatiuon(size, n=10, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(points=3, genMax=1000)
                print(f"Best eval: {best['eval']}, Generation: {gen}, popSize: {size}")

                x = arange(gen + 1)

                for xi in x:
                    f.write(
                        f"{xi}, {seed}, {size}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


def maxGenExperiment(howManyTimes=5):
    maxGenSizes = [10, 50, 100, 1000, 10000]
    path = "./csv/generations.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for _ in range(howManyTimes):
            seed = random.randint(0, 10000)
            for size in maxGenSizes:
                pop = Populatiuon(100, n=10, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(points=3, genMax=size)
                print(f"Best eval: {best['eval']}, Generation: {gen}, genMax: {size}")

                x = arange(gen + 1)

                for xi in x:
                    f.write(
                        f"{xi}, {seed}, {size}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


def pnExperiment(howManyTimes=5):
    pns = [0.01, 0.1, 0.2, 0.3, 0.5]
    path = "./csv/mutation.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for _ in range(howManyTimes):
            seed = random.randint(0, 10000)
            for chance in pns:
                pop = Populatiuon(100, n=10, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(
                    points=3, genMax=1000, pn=chance
                )
                print(
                    f"Best eval: {best['eval']}, Generation: {gen}, mutation chance: {chance}"
                )

                x = arange(gen + 1)

                for xi in x:
                    f.write(
                        f"{xi}, {seed}, {chance}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


def pcExperiment(howManyTimes=5):
    pcs = [0.3, 0.5, 0.7, 0.8, 0.9]
    path = "./csv/crossover.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for _ in range(howManyTimes):
            seed = random.randint(0, 10000)
            for chance in pcs:
                pop = Populatiuon(100, n=10, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(
                    points=3, genMax=1000, pc=chance
                )
                print(
                    f"Best eval: {best['eval']}, Generation: {gen}, crossover chance: {chance}"
                )

                for xi in range(gen + 1):
                    f.write(
                        f"{xi}, {seed}, {chance}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


def indivPerRoundExperiment(howManyTimes=5):
    tournament = [2, 3, 4, 5, 10]
    path = "./csv/tournament.csv"
    exists = Path(path).is_file()
    with open(path, "a") as f:
        if not exists:
            f.write("generation,run,parameter,mean_fitness,best_fitness\n")
        for _ in range(howManyTimes):
            seed = random.randint(0, 10000)
            for size in tournament:
                pop = Populatiuon(100, n=10, seed=seed)
                best, gen, meanFit, bestFit = pop.evolve(
                    points=3, genMax=1000, indivPerRound=size
                )
                print(
                    f"Best eval: {best['eval']}, Generation: {gen}, tournament size: {size}"
                )

                x = arange(gen + 1)

                for xi in x:
                    f.write(
                        f"{xi}, {seed}, {size}, {meanFit[int(xi)]}, {bestFit[int(xi)]}\n"
                    )


n = 5
baseExperiment(1)
print()
popSizeExperiment(n)
print()
pcExperiment(n)
print()
pnExperiment(n)
print()
maxGenExperiment(n)
print()
indivPerRoundExperiment(n)
