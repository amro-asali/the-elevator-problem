import math
import random
from matplotlib import pyplot as plt


def nCr(n, r):
    """Compute the binomial coefficient "n choose r" using the factorial function."""

    f = math.factorial
    return f(n) / (f(n - r) * f(r))


def algorithm1(M, K, i, j_list):
    """Implementation of algorithm 1 for approximate sorting.

        Args:
            M (int): The maximum value of the elements in the array.
            K (int): The length of the array.
            i (int): The current element being sorted.
            j_list (List[int]): A binary list representing the elements of the array that have already been sorted.

        Returns:
            int: The index at which the current element should be inserted in the sorted array.
    """
    # P(Q=j)=(iM)j-1(M-i+1M)K-j(K-1 choose j-1)
    N = K
    max_p = 0
    max_j = -1
    for j in range(N + 1):
        if not j_list[j]:  # j is unoccupied
            p1 = (i / M) ** (j - 1)  # probability for (j-1) tenants to live below or at i
            p2 = (((M - i + 1) / M) ** (K - j)) * nCr(K - 1, j - 1)  # probability for (k-j) tenants to live
            # above or at i
            p = p1 * p2
            if p > max_p:
                max_p = p
                max_j = j
    return max_j


def algorithm0(M, K, i, j_list):
    """Implementation of algorithm 0 for approximate sorting.

        Args:
            M (int): The maximum value of the elements in the array.
            K (int): The length of the array.
            i (int): The current element being sorted.
            j_list (List[int]): A binary list representing the elements of the array that have already been sorted.

        Returns:
            int: The index at which the current element should be inserted in the sorted array.
    """
    if i < K:
        if not j_list[i]:
            return i
        else:
            for j in range(1, max(i, K - i)):
                if i - j >= 0 and not j_list[i - j]:
                    return j
                elif i + j < len(j_list) and not j_list[i + j]:
                    return j

    return algorithm1(M, K, i, j_list)


def approXsort(M, K, arr, algorithm):
    """Sort an array approximately using the given algorithm.

        Args:
            M (int): The maximum value of the elements in the array.
            K (int): The length of the array.
            arr (List[int]): The array to be sorted.
            algorithm (Callable): The algorithm to be used for sorting.

        Returns:
            List[int]: The sorted array.
    """
    N = K
    j_list = [1] + [0] * N
    res = [-1] * N
    for k in range(K):
        T = arr[k]
        J = algorithm(M=M, K=K, i=T, j_list=j_list)
        j_list[J] = 1
        res[J - 1] = T

    return res


def plot(K_s, Y_s):
    """Plot the average number of tenants blocking the door comparison as a function of K.

        Args:
            K_s (list): A list of integers representing the values of K for which the experiment was run.
            Y_s (list): A list of three lists, each containing the average number of tenants blocking the door for the
                        original, approximate sorted algorithm 0, and approximate sorted algorithm 1, respectively.

        Returns:
            None.
    """
    plt.title('average number of tenants blocking the door comparison as a function of K')
    plt.xlabel('K')
    plt.ylabel('avg #tenants blocking the door')
    plt.scatter(K_s, Y_s[0], label='original')
    plt.scatter(K_s, Y_s[1], label='approx sorted algo0', c='RED')
    plt.scatter(K_s, Y_s[2], label='approx sorted algo1', c='BLACK')
    plt.legend()
    plt.show()


def compare_algorithms():
    """Compare the original algorithm to two approximate sorted algorithms by running experiments for different values
        of K and plotting the results.

        Args:
            None.

        Returns:
            None.
    """
    num_blocking_all = [[], [], []]  # [ org , algo0 , algo1 ]
    M = 100
    K_s = sorted(set([int(random.uniform(2, 100)) for _ in range(50)]))
    for K in K_s:
        original = [int(random.uniform(1, M)) for _ in range(K)]
        res0 = approXsort(M=M, K=K, arr=original, algorithm=algorithm0)
        res1 = approXsort(M=M, K=K, arr=original, algorithm=algorithm1)
        blocking_algo0, blocking_algo1, blocking_original = 0, 0, 0
        for i in range(K):
            for j in range(i):
                if res0[j] > res0[i]:
                    blocking_algo0 += 1
                if original[j] > original[i]:
                    blocking_original += 1
                if res1[j] > res1[i]:
                    blocking_algo1 += 1
        num_blocking_all[0].append(blocking_original / K)
        num_blocking_all[1].append(blocking_algo0 / K)
        num_blocking_all[2].append(blocking_algo1 / K)
    plot(K_s, num_blocking_all)


if __name__ == '__main__':
    compare_algorithms()
