import math
import random

from matplotlib import pyplot as plt


def nCr(n, r):
    f = math.factorial
    return f(n) / (f(n - r) * f(r))


def algorithm1(M, K, i, j_list):
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
    N = K
    j_list = [1] + [0] * (N)
    res = [-1] * N
    for k in range(K):
        T = arr[k]
        J = algorithm(M=M, K=K, i=T, j_list=j_list)
        j_list[J] = 1
        res[J - 1] = T

    return res


def plot(K_s,Y_s ):
    plt.title('average number of tenants blocking the door comparison as a function of K')
    plt.xlabel('K')
    plt.ylabel('avg #tenants blocking the door')
    plt.scatter(K_s, Y_s[1], label='approx sorted algo0', c='RED')
    plt.scatter(K_s, Y_s[2], label='approx sorted algo1', c='BLACK')
    plt.scatter(K_s, Y_s[0], label='original')
    plt.legend()
    plt.show()


def compare_algorithms():
    num_blocking_all = [[], [], []]  # [ org , algo0 , algo1 ]
    M = 100
    K_s = sorted(set([int(random.uniform(2, 100)) for i in range(50)]))
    for K in K_s:
        original = [int(random.uniform(1, M)) for i in range(K)]
        res0 = approXsort(M=M, K=len(original), arr=original, algorithm=algorithm0)
        res1 = approXsort(M=M, K=len(original), arr=original, algorithm=algorithm1)
        count_sorted_algo0 = 0
        count_sorted_algo1 = 0
        count_original = 0
        for i in range(len(res0)):
            for j in range(i + 1, len(res0)):
                if res0[i] > res0[j]:
                    count_sorted_algo0 += 1
                if original[i] > original[j]:
                    count_original += 1
                if res1[i] > res1[j]:
                    count_sorted_algo1 += 1
        num_blocking_all[0].append(count_original / K)
        num_blocking_all[1].append(count_sorted_algo0 / K)
        num_blocking_all[2].append(count_sorted_algo1 / K)
    plot(K_s,num_blocking_all)


if __name__ == '__main__':
    compare_algorithms()
