"""
HMM utilities for the Fishing Derby (pure Python, no numpy, so it runs fast under PyPy).
Based on kattis/hmm3.py, extended to train on SEVERAL sequences at once.
A model is a tuple (A, B, pi).
"""
import math
import random

B_FLOOR = 1e-4   # smallest allowed emission probability (smoothing, see notes)


def random_model(n_states, n_emissions, rng):
    """Near-uniform random model (+/-30% noise): breaks symmetry, avoids exact zeros (C experiment 2)."""
    def row(n):
        r = [(1.0 + rng.uniform(-0.3, 0.3)) / n for _ in range(n)]
        s = sum(r)
        return [x / s for x in r]
    A = [row(n_states) for _ in range(n_states)]
    B = [row(n_emissions) for _ in range(n_states)]
    pi = row(n_states)
    return A, B, pi


def forward_scaled(A, B, pi, obs):
    """Scaled forward pass. Returns (alpha, c) with c[t] = 1 / sum_i alpha_t(i)."""
    N, T = len(A), len(obs)
    alpha = [[0.0] * N for _ in range(T)]
    c = [0.0] * T
    o = obs[0]
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][o]
    s = sum(alpha[0])
    c[0] = 1.0 / s if s > 0.0 else 1e300
    for i in range(N):
        alpha[0][i] *= c[0]
    for t in range(1, T):
        o = obs[t]
        prev = alpha[t - 1]
        cur = alpha[t]
        for i in range(N):
            acc = 0.0
            for j in range(N):
                acc += prev[j] * A[j][i]
            cur[i] = acc * B[i][o]
        s = sum(cur)
        c[t] = 1.0 / s if s > 0.0 else 1e300
        for i in range(N):
            cur[i] *= c[t]
    return alpha, c


def log_likelihood(model, obs):
    """log P(obs | model) = -sum_t log c_t."""
    A, B, pi = model
    _, c = forward_scaled(A, B, pi, obs)
    return -sum(math.log(x) for x in c)


def backward_scaled(A, B, obs, c):
    N, T = len(A), len(obs)
    beta = [[0.0] * N for _ in range(T)]
    for i in range(N):
        beta[T - 1][i] = c[T - 1]
    for t in range(T - 2, -1, -1):
        o = obs[t + 1]
        nxt = beta[t + 1]
        for i in range(N):
            acc = 0.0
            Ai = A[i]
            for j in range(N):
                w = Ai[j] * B[j][o]
                if w != 0.0:
                    acc += w * nxt[j]
            beta[t][i] = c[t] * acc
    return beta


def baum_welch_multi(model, seqs, n_emissions, max_iters=30, tol=1e-3):
    """
    Baum-Welch on several independent sequences: expected counts are summed over sequences,
    then normalised once (instead of concatenating the sequences).
    """
    A, B, pi = model
    N = len(A)
    old = float('-inf')
    for _ in range(max_iters):
        trans = [[0.0] * N for _ in range(N)]
        from_i = [0.0] * N
        emit = [[0.0] * n_emissions for _ in range(N)]
        in_i = [0.0] * N
        pi_acc = [0.0] * N
        total = 0.0
        for obs in seqs:
            T = len(obs)
            alpha, c = forward_scaled(A, B, pi, obs)
            beta = backward_scaled(A, B, obs, c)
            total += -sum(math.log(x) for x in c)
            for t in range(T - 1):
                o_next = obs[t + 1]
                a_t = alpha[t]
                b_next = beta[t + 1]
                o_t = obs[t]
                for i in range(N):
                    ai = a_t[i]
                    if ai == 0.0:
                        continue
                    gi = 0.0
                    Ai = A[i]
                    tr = trans[i]
                    for j in range(N):
                        g = ai * Ai[j] * B[j][o_next] * b_next[j]
                        tr[j] += g
                        gi += g
                    from_i[i] += gi
                    in_i[i] += gi
                    emit[i][o_t] += gi
                    if t == 0:
                        pi_acc[i] += gi
            for i in range(N):
                g = alpha[T - 1][i]
                in_i[i] += g
                emit[i][obs[T - 1]] += g
        if not (total - old >= tol):
            break
        old = total
        A = [[trans[i][j] / from_i[i] if from_i[i] > 0 else A[i][j] for j in range(N)] for i in range(N)]
        newB = []
        for i in range(N):
            row = [max(emit[i][k] / in_i[i], B_FLOOR) if in_i[i] > 0 else B[i][k] for k in range(n_emissions)]
            s = sum(row)
            newB.append([x / s for x in row])
        B = newB
        s = sum(pi_acc)
        pi = [x / s for x in pi_acc] if s > 0 else pi
    return A, B, pi
