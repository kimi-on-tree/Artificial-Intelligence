"""
C-level empirical investigation (DD2380 Lab2).
Uses the Baum-Welch implementation in kattis/hmm3.py (forward_scaled, baum_welch_step).
Run from the Lab2-HMM folder:  python kattis/c_experiments.py
Writes results to c_results.txt
"""
import math, random, itertools, time, sys, os
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hmm3 import forward_scaled, baum_welch_step

ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "kattis" else HERE
TOL = 1e-6          # same stopping rule for every run: improvement in total log-likelihood
MAX_ITERS = 3000    # cap (larger than the Kattis cap of 500, to observe true convergence)

# ---------- generating model (from the PDF) ----------
A_TRUE = [[0.70, 0.05, 0.25], [0.10, 0.80, 0.10], [0.20, 0.30, 0.50]]
B_TRUE = [[0.70, 0.20, 0.10, 0.00], [0.10, 0.40, 0.30, 0.20], [0.00, 0.10, 0.20, 0.70]]
PI_TRUE = [1.0, 0.0, 0.0]

# ---------- initialisations ----------
A0 = [[0.54, 0.26, 0.20], [0.19, 0.53, 0.28], [0.22, 0.18, 0.60]]
B0 = [[0.50, 0.20, 0.11, 0.19], [0.22, 0.28, 0.23, 0.27], [0.19, 0.21, 0.15, 0.45]]
PI0 = [0.3, 0.2, 0.5]

INITS = {
    "A0/B0/pi0 (given)": (A0, B0, PI0),
    "uniform": ([[1/3]*3 for _ in range(3)], [[0.25]*4 for _ in range(3)], [1/3]*3),
    "diagonal A, pi=(0,0,1)": ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], B0, [0.0, 0.0, 1.0]),
    "near-true, non-symmetric": (
        [[0.60, 0.10, 0.30], [0.15, 0.70, 0.15], [0.25, 0.25, 0.50]],
        [[0.60, 0.25, 0.10, 0.05], [0.10, 0.35, 0.35, 0.20], [0.05, 0.10, 0.25, 0.60]],
        [0.90, 0.05, 0.05]),
}


def load(name):
    tok = open(os.path.join(ROOT, name)).read().split()
    T = int(tok[0])
    return [int(x) for x in tok[1:1 + T]]


def loglik(A, B, pi, obs):
    _, c = forward_scaled(A, B, pi, obs)
    return -sum(math.log(x) for x in c)


def train(A, B, pi, obs, tol=TOL, max_iters=MAX_ITERS):
    """Same rule as hmm3.baum_welch, but also returns the iteration count and history."""
    old = float('-inf')
    hist = []
    for it in range(1, max_iters + 1):
        nA, nB, npi, lp = baum_welch_step(A, B, pi, obs)
        hist.append(lp)
        if not (lp - old >= tol):   # also stops on NaN
            return A, B, pi, it - 1, old, hist, True
        A, B, pi, old = nA, nB, npi, lp
    return A, B, pi, max_iters, loglik(A, B, pi, obs), hist, False


def rand_stochastic(rows, cols, rng):
    """Near-uniform random row-stochastic matrix: 1/cols +/- 30%, then normalised."""
    M = []
    for _ in range(rows):
        r = [1.0 / cols * (1 + rng.uniform(-0.3, 0.3)) for _ in range(cols)]
        s = sum(r)
        M.append([x / s for x in r])
    return M


def best_perm(A, B):
    """Permutation of learned state labels that best matches the generating model."""
    best = None
    for p in itertools.permutations(range(3)):
        err = sum(abs(A[p[i]][p[j]] - A_TRUE[i][j]) for i in range(3) for j in range(3)) \
            + sum(abs(B[p[i]][k] - B_TRUE[i][k]) for i in range(3) for k in range(4))
        if best is None or err < best[0]:
            best = (err, p)
    return best


def permute(A, B, pi, p):
    n = len(p)
    return ([[A[p[i]][p[j]] for j in range(n)] for i in range(n)],
            [B[p[i]][:] for i in range(n)], [pi[p[i]] for i in range(n)])


def fmt(M):
    return "\n".join("    [" + ", ".join(f"{x:.3f}" for x in row) + "]" for row in M)


# ---------- jobs ----------
def job(task):
    kind = task[0]
    t0 = time.time()
    if kind in ("exp1", "exp2"):
        _, name, data, init = task
        obs = load(data)
        A, B, pi = INITS[init]
        A, B, pi, it, lp, hist, conv = train([r[:] for r in A], [r[:] for r in B], pi[:], obs)
        return dict(kind=kind, name=name, data=data, init=init, A=A, B=B, pi=pi, iters=it,
                    lp=lp, T=len(obs), conv=conv, hist=hist, secs=time.time() - t0)
    if kind == "exp3":
        _, N, r = task
        obs = load("hmm_c_N10000.in")
        rng = random.Random(1000 * N + r)          # seed = 1000*N + restart index
        A = rand_stochastic(N, N, rng)
        B = rand_stochastic(N, 4, rng)
        pi = rand_stochastic(1, N, rng)[0]
        A, B, pi, it, lp, hist, conv = train(A, B, pi, obs)
        ev = loglik(A, B, pi, load("hmm_c_N1000.in"))
        return dict(kind=kind, N=N, r=r, A=A, B=B, pi=pi, iters=it, lp=lp, T=len(obs),
                    ev=ev, conv=conv, secs=time.time() - t0)


def main():
    tasks = [("exp1", "N1000", "hmm_c_N1000.in", "A0/B0/pi0 (given)"),
             ("exp1", "N10000", "hmm_c_N10000.in", "A0/B0/pi0 (given)")]
    for init in ["uniform", "diagonal A, pi=(0,0,1)", "near-true, non-symmetric"]:
        tasks.append(("exp2", init, "hmm_c_N10000.in", init))
    RESTARTS = 3
    for N in (2, 3, 4):
        for r in range(RESTARTS):
            tasks.append(("exp3", N, r))
    with Pool(2) as pool:
        res = pool.map(job, tasks)

    o1000, o10000 = load("hmm_c_N1000.in"), load("hmm_c_N10000.in")
    out = []
    P = out.append
    P(f"Stopping rule: stop when log P improves by < {TOL} (total, not per observation), "
      f"or after {MAX_ITERS} iterations.\n")
    P("Reference: generating model log P / T")
    P(f"  N1000 : {loglik(A_TRUE, B_TRUE, PI_TRUE, o1000)/len(o1000):.5f}")
    P(f"  N10000: {loglik(A_TRUE, B_TRUE, PI_TRUE, o10000)/len(o10000):.5f}\n")

    P("=== Experiment 1: sequence length (init A0/B0/pi0) ===")
    for r in [x for x in res if x["kind"] == "exp1"]:
        err, p = best_perm(r["A"], r["B"])
        A, B, pi = permute(r["A"], r["B"], r["pi"], p)
        P(f"\n[{r['name']}] iterations={r['iters']} converged={r['conv']}  "
          f"logP/T={r['lp']/r['T']:.5f}  time={r['secs']:.1f}s")
        h = r["hist"]
        P("  logP/T at iter 1,10,100,500,last: " + ", ".join(
            f"{h[min(k, len(h)) - 1]/r['T']:.5f}" for k in (1, 10, 100, 500, len(h))))
        P(f"  state permutation={p}  sum|learned-true| (A+B) = {err:.3f}")
        P("  A =\n" + fmt(A)); P("  B =\n" + fmt(B)); P("  pi = " + str([round(x, 3) for x in pi]))

    P("\n=== Experiment 2: initialisation (trained on N10000) ===")
    e1 = [x for x in res if x["kind"] == "exp1" and x["name"] == "N10000"][0]
    for r in [dict(e1, name="A0/B0/pi0 (given)")] + [x for x in res if x["kind"] == "exp2"]:
        err, p = best_perm(r["A"], r["B"])
        P(f"\n[{r['name']}] iterations={r['iters']} converged={r['conv']}  "
          f"logP/T={r['lp']/r['T']:.5f}  sum|learned-true|={err:.3f}")
        P("  A =\n" + fmt(r["A"])); P("  B =\n" + fmt(r["B"]))
        P("  pi = " + str([round(x, 3) for x in r["pi"]]))

    P("\n=== Experiment 3: number of hidden states (train N10000, evaluate N1000) ===")
    P("Init: near-uniform random (each entry 1/n * (1 +/- U(0.3)), normalised), "
      "seed = 1000*N + restart, 3 restarts, keep highest TRAINING log-likelihood.")
    P(f"{'N':>2} {'#free params':>12} {'train logP/T':>13} {'eval logP/T':>12}  restarts (train / eval / iters)")
    for N in (2, 3, 4):
        runs = [x for x in res if x["kind"] == "exp3" and x["N"] == N]
        best = max(runs, key=lambda x: x["lp"])
        k = N * (N - 1) + N * 3 + (N - 1)
        detail = "; ".join(f"{x['lp']/x['T']:.5f} / {x['ev']/len(o1000):.5f} / {x['iters']}" for x in runs)
        P(f"{N:>2} {k:>12} {best['lp']/best['T']:>13.5f} {best['ev']/len(o1000):>12.5f}  {detail}")
        if N == 3:
            err, p = best_perm(best["A"], best["B"])
            A, B, pi = permute(best["A"], best["B"], best["pi"], p)
            P(f"   3-state best model after permutation {p}, sum|learned-true|={err:.3f}")
            P("   A =\n" + fmt(A)); P("   B =\n" + fmt(B))
    text = "\n".join(out)
    print(text)
    open(os.path.join(ROOT, "c_results.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
