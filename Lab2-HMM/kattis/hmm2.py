import sys
import math


def read_matrix(tokens, pos):
    rows = int(tokens[pos])
    cols = int(tokens[pos + 1])
    pos += 2
    M = []
    for i in range(rows):
        M.append([float(x) for x in tokens[pos:pos + cols]])
        pos += cols
    return M, pos


def read_sequence(tokens, pos):
    T = int(tokens[pos])
    pos += 1
    seq = [int(x) for x in tokens[pos:pos + T]]
    pos += T
    return seq, pos


def safe_log(x):
    return math.log(x) if x > 0 else float('-inf')


def viterbi(A, B, pi, obs):
    N = len(A)
    T = len(obs)

    logA = [[safe_log(A[i][j]) for j in range(N)] for i in range(N)]
    logB = [[safe_log(B[i][k]) for k in range(len(B[0]))] for i in range(N)]

    # t = 1: delta_1(i) = log pi_i + log b_i(o_1)
    delta = [safe_log(pi[i]) + logB[i][obs[0]] for i in range(N)]
    psi = [[0] * N] 

    # t = 2..T
    for t in range(1, T):
        o = obs[t]
        new_delta = []
        back = []
        for i in range(N):
            best_j = 0
            best_val = delta[0] + logA[0][i]
            for j in range(1, N):
                val = delta[j] + logA[j][i]
                if val > best_val:
                    best_val = val
                    best_j = j
            new_delta.append(best_val + logB[i][o])
            back.append(best_j)
        delta = new_delta
        psi.append(back)

    last = 0
    for i in range(1, N):
        if delta[i] > delta[last]:
            last = i

    # x_t = psi_{t+1}(x_{t+1})
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(psi[t][path[-1]])
    path.reverse()
    return path


def main():
    tokens = sys.stdin.read().split()
    A, pos = read_matrix(tokens, 0)
    B, pos = read_matrix(tokens, pos)
    pi, pos = read_matrix(tokens, pos)
    obs, pos = read_sequence(tokens, pos)

    path = viterbi(A, B, pi[0], obs)
    print(" ".join(str(s) for s in path))


main()
