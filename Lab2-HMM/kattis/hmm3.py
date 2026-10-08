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


def forward_scaled(A, B, pi, obs):
    N, T = len(A), len(obs)
    alpha = [[0.0] * N for _ in range(T)]
    c = [0.0] * T

    # t = 0
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][obs[0]]
    c[0] = 1.0 / sum(alpha[0])
    for i in range(N):
        alpha[0][i] *= c[0]

    # t = 1..T-1
    for t in range(1, T):
        o = obs[t]
        prev = alpha[t - 1]
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += prev[j] * A[j][i]
            alpha[t][i] = s * B[i][o]
        c[t] = 1.0 / sum(alpha[t])
        for i in range(N):
            alpha[t][i] *= c[t]
    return alpha, c


def backward_scaled(A, B, obs, c):
    N, T = len(A), len(obs)
    beta = [[0.0] * N for _ in range(T)]
    for i in range(N):
        beta[T - 1][i] = c[T - 1]

    for t in range(T - 2, -1, -1):
        o = obs[t + 1]
        nxt = beta[t + 1]
        for i in range(N):
            s = 0.0
            for j in range(N):
                w = A[i][j] * B[j][o]
                if w != 0.0:      
                    s += w * nxt[j]
            beta[t][i] = c[t] * s
    return beta


def baum_welch_step(A, B, pi, obs):
    N, K, T = len(A), len(B[0]), len(obs)
    alpha, c = forward_scaled(A, B, pi, obs)
    beta = backward_scaled(A, B, obs, c)

    # log P(O | lambda) = -sum_t log c_t
    log_prob = -sum(math.log(ct) for ct in c)

    # E：
    trans_num = [[0.0] * N for _ in range(N)]
    gamma_sum_1 = [0.0] * N
    emit_num = [[0.0] * K for _ in range(N)]
    gamma_sum_all = [0.0] * N
    new_pi = [0.0] * N

    for t in range(T - 1):
        o_next = obs[t + 1]
        a_t = alpha[t]
        b_next = beta[t + 1]
        for i in range(N):
            if a_t[i] == 0.0:
                continue
            gi = 0.0
            Ai = A[i]
            for j in range(N):
                if Ai[j] == 0.0:     
                    continue
                g = a_t[i] * Ai[j] * B[j][o_next] * b_next[j]
                trans_num[i][j] += g
                gi += g
            gamma_sum_1[i] += gi
            gamma_sum_all[i] += gi
            emit_num[i][obs[t]] += gi
            if t == 0:
                new_pi[i] = gi

    # gamma_{T-1}(i) = alpha_{T-1}(i)
    for i in range(N):
        g = alpha[T - 1][i]
        gamma_sum_all[i] += g
        emit_num[i][obs[T - 1]] += g

    # M
    new_A = [[trans_num[i][j] / gamma_sum_1[i] if gamma_sum_1[i] > 0 else A[i][j]
              for j in range(N)] for i in range(N)]
    new_B = [[emit_num[i][k] / gamma_sum_all[i] if gamma_sum_all[i] > 0 else B[i][k]
              for k in range(K)] for i in range(N)]
    return new_A, new_B, new_pi, log_prob


def baum_welch(A, B, pi, obs, max_iters=500, tol=1e-6):
    old_log_prob = float('-inf')
    it = 0
    for it in range(1, max_iters + 1):
        new_A, new_B, new_pi, log_prob = baum_welch_step(A, B, pi, obs)
        if not (log_prob - old_log_prob >= tol):
            break
        A, B, pi = new_A, new_B, new_pi
        old_log_prob = log_prob
    return A, B, pi, it, old_log_prob


def format_matrix(M):
    return str(len(M)) + " " + str(len(M[0])) + " " + " ".join(
        str(round(x, 6)) for row in M for x in row)


def main():
    tokens = sys.stdin.read().split()
    A, pos = read_matrix(tokens, 0)
    B, pos = read_matrix(tokens, pos)
    pi, pos = read_matrix(tokens, pos)
    obs, pos = read_sequence(tokens, pos)

    A, B, pi, iters, log_prob = baum_welch(A, B, pi[0], obs)
    print(format_matrix(A))
    print(format_matrix(B))
    print("iterations:", iters, "logP/T:", log_prob / len(obs), file=sys.stderr)


if __name__ == "__main__":
    main()
