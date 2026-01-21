# pyright: standard
import random
import time
from typing import Dict, List


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

    def cross(self, other):
        idx = random.randint(1, self._size - 2)

        child1 = BoardVector(self._size)
        child2 = BoardVector(self._size)

        child1._vector = self._vector[:idx] + other._vector[idx:]
        child2._vector = other._vector[:idx] + self._vector[idx:]

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
    def __init__(self, popSize=100, n=6):
        self._popSize = popSize
        self._pop: List[Dict] = []
        self._n = n

        for _ in range(self._popSize):
            bv = BoardVector(n)
            self._pop.append({"board": bv, "eval": bv.conflicts()})

    def evaluate(self, pop):
        for i, indv in enumerate(pop):
            indv["eval"] = indv["board"].conflicts()

    def crossover(self, P: List[Dict], pc: float):
        Pn = []
        for i in range(0, len(P), 2):
            if random.random() < pc:
                child1, child2 = P[i]["board"].cross(P[i + 1]["board"])
                Pn.append({"board": child1, "eval": 0})
                Pn.append({"board": child2, "eval": 0})
            else:
                Pn.append(P[i])
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

    def evolve(self, pn=0.2, pc=0.8, genMax=10000, indivPerRound=3):
        gen = 0
        _, best = self.bestCandidate(self._pop)
        while gen < genMax and best["eval"] > 0:
            Pn = self.selection(self._pop, indivPerRound)
            Pn = self.crossover(Pn, pc)
            Pn = self.mutation(Pn, pn)
            self.evaluate(Pn)

            _, cand = self.bestCandidate(Pn)
            if cand["eval"] < best["eval"]:
                best = cand
            self.replacement(Pn, best)
            self.evaluate(self._pop)
            gen += 1

        return best, gen

    def printPopulation(self):
        for indv in self._pop:
            print(f"{indv['eval']}: ", end="")
            indv["board"].printBoard()


pop = Populatiuon(20, 10)

pop.printPopulation()
best, gen = pop.evolve(indivPerRound=3)
print()
print(gen)
pop.printPopulation()

print(f"{best['eval']}: ", end="")
print(best["board"].conflicts())
