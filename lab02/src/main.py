import random
import time
from collections import deque


class BoardPerm:
    def __init__(self, size=0, parent=None):
        self._size = size
        self._perm = []
        self._parent = parent

    def generate_board(self, size):
        self._size = size
        self._perm = list(range(self._size))
        random.shuffle(self._perm)

    def makeBoard(self, array, parent=None):
        self._perm = list(array)
        self._size = len(array)
        self._parent = parent

    def printBoard(self):
        matrix = [[0] * self._size for _ in range(self._size)]
        for i in range(self._size):
            matrix[i][self._perm[i]] = 1
        for row in matrix:
            print(row)

    def printPerm(self):
        print(self._perm)

    def conflicts(self):
        for i in range(self._size):
            for j in range(i + 1, self._size):
                if abs(self._perm[j] - self._perm[i]) == abs(j - i):
                    return True
        return False

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

    @staticmethod
    def BFS(size):
        opened = deque([BoardPerm()])
        closed = set()
        openCount = 1
        closeCount = 0

        while opened:
            s = opened.popleft()

            if s._size == size and not s.conflicts():
                return s, openCount, closeCount

            ts = BoardPerm.generateChildren(s, size)
            for t in ts:
                if not t.conflicts() and tuple(t._perm) not in closed:
                    opened.append(t)
                    openCount += 1
            key = tuple(s._perm)
            closed.add(key)
            closeCount += 1

        return None, openCount, closeCount

    @staticmethod
    def DFS(size):
        opened = deque([BoardPerm()])
        closed = set()
        openCount = 1
        closeCount = 0

        while opened:
            s = opened.pop()

            if s._size == size and not s.conflicts():
                return s, openCount, closeCount

            ts = BoardPerm.generateChildren(s, size)
            for t in ts:
                if not t.conflicts() and tuple(t._perm) not in closed:
                    opened.append(t)
                    openCount += 1
            key = tuple(s._perm)
            closed.add(key)
            closeCount += 1

        return None, openCount, closeCount

    @staticmethod
    def generateChildren(parent, size):
        children = []
        for row in range(size):
            if row in parent._perm:
                continue
            t = parent._perm + [row]
            b = BoardPerm()
            b.makeBoard(t, parent=parent)
            children.append(b)
        return children


class BoardMatrix:
    def __init__(self, size=0, parent=None):
        self._size = size
        self._matrix = [[0] * self._size for _ in range(self._size)]
        self._parent = parent

    def generate_board(self, size):
        self._size = size
        self._matrix = [[0] * self._size for _ in range(self._size)]
        for col in range(self._size):
            row = random.randint(0, self._size - 1)
            self._matrix[row][col] = 1

    def makeBoard(self, matrix, parent=None):
        self._matrix = [list(row) for row in matrix]
        self._size = len(matrix)
        self._parent = parent

    def printBoard(self):
        for row in self._matrix:
            print(row)

    def conflicts(self):
        for i in range(self._size):
            row_count = sum(self._matrix[i])
            col_count = sum(self._matrix[j][i] for j in range(self._size))
            if row_count > 1 or col_count > 1:
                return True

        for row in range(self._size):
            for col in range(self._size):
                if self._matrix[row][col] != 1:
                    continue
                for row2 in range(row + 1, self._size):
                    for col2 in range(self._size):
                        if self._matrix[row2][col2] != 1:
                            continue
                        if abs(row - row2) == abs(col - col2):
                            return True
        return False

    def conflictsOptimized(self):
        for i in range(self._size):
            row_count = sum(self._matrix[i])
            col_count = sum(self._matrix[j][i] for j in range(self._size))
            if row_count > 1 or col_count > 1:
                return True

        for row in range(self._size):
            for col in range(self._size):
                if self._matrix[row][col] != 1:
                    continue
                for i in range(1, self._size):
                    if (
                        row + i < self._size
                        and col + i < self._size
                        and self._matrix[row + i][col + i] == 1
                    ):
                        return True
                    if (
                        row + i < self._size
                        and col - i >= 0
                        and self._matrix[row + i][col - i] == 1
                    ):
                        return True
                    if (
                        row - i >= 0
                        and col + i < self._size
                        and self._matrix[row - i][col + i] == 1
                    ):
                        return True
                    if (
                        row - i >= 0
                        and col - i >= 0
                        and self._matrix[row - i][col - i] == 1
                    ):
                        return True
        return False

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

    def measure_time_optimized(self, iterations):
        cum_time = 0
        conflict_counter = 0
        for _ in range(iterations):
            start = time.perf_counter()
            if self.conflictsOptimized():
                conflict_counter += 1
            end = time.perf_counter()
            cum_time += end - start
        conflict_rate = conflict_counter / iterations
        avg = cum_time / iterations
        return avg, conflict_rate

    @staticmethod
    def BFS(size):
        start = BoardMatrix()
        start.makeBoard([[0] * size for _ in range(size)])
        opened = deque([start])
        closed = set()
        openCount = 1
        closeCount = 0

        while opened:
            s = opened.popleft()
            queenCount = sum([sum(col) for col in s._matrix])

            if queenCount == size and not s.conflictsOptimized():
                return s, openCount, closeCount

            if queenCount < size:
                ts = BoardMatrix.generateChildren(s)
                for t in ts:
                    key = tuple(tuple(row) for row in t._matrix)
                    if not t.conflictsOptimized() and t._matrix not in opened and key not in closed:
                        opened.append(t)
                        openCount += 1
            key = tuple(tuple(row) for row in s._matrix)
            closed.add(key)
            closeCount += 1

        return None, openCount, closeCount

    @staticmethod
    def DFS(size):
        start = BoardMatrix()
        start.makeBoard([[0] * size for _ in range(size)])
        opened = deque([start])
        closed = set()
        openCount = 1
        closeCount = 0

        while opened:
            s = opened.pop()
            queenCount = sum([sum(col) for col in s._matrix])

            if queenCount == size and not s.conflictsOptimized():
                return s, openCount, closeCount

            if queenCount < size:
                ts = BoardMatrix.generateChildren(s)
                for t in ts:
                    key = tuple(tuple(row) for row in t._matrix)
                    if not t.conflictsOptimized() and t._matrix not in opened and key not in closed:
                        opened.append(t)
                        openCount += 1
            key = tuple(tuple(row) for row in s._matrix)
            closed.add(key)
            closeCount += 1

        return None, openCount, closeCount

    @staticmethod
    def generateChildren(parent):
        size = parent._size
        children = []
        for row in range(size):
            for col in range(size):
                new_matrix = [list(r) for r in parent._matrix]
                if new_matrix[row][col] == 1:
                    continue
                new_matrix[row][col] = 1
                b = BoardMatrix()
                b.makeBoard(new_matrix, parent=parent)
                children.append(b)
        return children