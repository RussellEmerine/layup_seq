import timeit
import math

import matplotlib.pyplot as plt


def layup_seq_naive(n: int) -> int:
    """
    A naive, *very* slow implementation of the Layup Sequence.
    """
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif n % 2 == 0:
        return layup_seq_naive(n - 1) + layup_seq_naive(n - 2)
    else:
        return 2 * layup_seq_naive(n - 1) - layup_seq_naive(n - 2)


def layup_seq_iterative(n: int) -> int:
    """
    A more efficient implementation of the Layup Sequence.

    This avoids recomputing S(n) for the same value of n.
    """
    if n == 1:
        return 1
    # a, b = S(i), S(i + 1) at the *end* of the loop
    a, b = 1, 2
    for i in range(2, n):
        # a, b = S(i - 1), S(i)
        if i % 2 == 0:
            # set a, b = S(i), S(i + 1): here i + 1 is odd
            a, b = b, 2 * b - a
        else:
            # set a, b = S(i), S(i + 1): here i + 1 is even
            a, b = b, b + a
    # at the end of the i = n - 1 loop, b = S(n)
    return b


class Vec:
    """
    A very simple 2D vector.
    Importing a library is also an option.
    """

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, item: int) -> int:
        return [self.x, self.y][item]


class Mat:
    """
    A very simple 2D matrix.
    Importing a library is also an option.
    """

    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        """
        Create the matrix [[a, b], [c, d]]
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __matmul__(self, other: 'Mat') -> 'Mat':
        """
        Multiply two matrices.
        """
        # precedence for the string literal hack: https://stackoverflow.com/a/36286947
        return Mat(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def __mul__(self, v: Vec) -> Vec:
        """
        Multiply a matrix onto a vector.
        """
        return Vec(self.a * v.x + self.b * v.y, self.c * v.x + self.d * v.y)


def layup_seq_binary(n: int) -> int:
    """
    A very efficient implementation of the Layup Sequence.

    Consider V(k) = [S(2k + 1), S(2k + 2)] as a 2D vector.
    Then, V(k + 1) = [S(2k + 3), S(2k + 4)]
    = [2 * S(2k + 2) - S(2k + 1), S(2k + 3) + S(2k + 2)]
    = [2 * S(2k + 2) - S(2k + 1), 3 * S(2k + 2) - S(2k + 1)]
    = M V(k)
    where M is the matrix [[-1, 2], [-1, 3]].
    Then, since V(0) = [1, 2] and V(k) = M^k V(0) we can evaluate
    using binary exponentiation on the matrix.
    """
    k = (n - 1) // 2

    # the M mentioned in the docstring
    M = Mat(-1, 2, -1, 3)
    # P = M^(2^i) at the *start* of the loop
    P = M
    # Q = M^(k & (2^(i + 1) - 1)), i.e. the i least significant bits of k
    Q = Mat(1, 0, 0, 1)

    for i in range(int(math.log2(k)) + 20):
        if (1 << i) & k:
            Q = Q @ P
        if (1 << i) > k:
            break
        P = P @ P

    V = Q * Vec(1, 2)
    return V[(n - 1) % 2]


if __name__ == '__main__':
    print("A simple test to verify that all the functions have the same behavior:")
    print("naive: ", layup_seq_naive(10))
    print("iterative: ", layup_seq_iterative(10))
    print("iterative: ", layup_seq_iterative(100))
    print("iterative: ", layup_seq_iterative(1000))
    print("iterative: ", layup_seq_iterative(10000))
    print("binary: ", layup_seq_binary(10))
    print("binary: ", layup_seq_binary(100))
    print("binary: ", layup_seq_binary(1000))
    print("binary: ", layup_seq_binary(10000))

    naive_xs = list(range(2, 21, 2))
    plt.plot(naive_xs, [
        timeit.timeit(f'layup_seq_naive({x})', setup="from __main__ import layup_seq_naive", number=1000)
        for x in naive_xs
    ])
    plt.xlabel('n')
    plt.xlim(0, 20)
    plt.ylabel('Runtime (s)')
    plt.savefig('naive.png')
    plt.close()

    iterative_xs = list(range(1000, 10001, 1000))
    plt.plot(iterative_xs, [
        timeit.timeit(f'layup_seq_iterative({x})', setup="from __main__ import layup_seq_iterative", number=1000)
        for x in iterative_xs
    ])
    plt.xlabel('n')
    plt.xlim(0, 10000)
    plt.ylabel('Runtime (s)')
    plt.savefig('iterative.png')
    plt.close()

    binary_xs = list(range(1000, 10001, 1000))
    plt.plot(binary_xs, [
        timeit.timeit(f'layup_seq_binary({x})', setup="from __main__ import layup_seq_binary", number=1000)
        for x in binary_xs
    ])
    plt.xlabel('n')
    plt.xlim(0, 10000)
    plt.ylabel('Runtime (s)')
    plt.savefig('binary.png')
    plt.close()
