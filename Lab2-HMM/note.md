## J2

### Q1: 

State order $(c_1, c_2)$, observation order $(H, T)$:

$$ \pi = \begin{pmatrix} 0.5 & 0.5 \end{pmatrix}, \quad A = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}, \quad B = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix} $$

- $\pi_1 = 0.5$: the probability that coin $c_1$ is used for the very first toss. 

- $a_{12} = 0.5$: if coin $c_1$ is used at time $t$, the probability of switching to coin $c_2$ at time $t+1$. 
- $b_1(T) = 0.1$: the probability that coin $c_1$ produces tails; since $c_1$ is biased towards heads, tails is rare.

### Q2：

With $p_t = (0.8, 0.2)$: $$ p_tA = (0.5,\ 0.5), \qquad p_tAB = (0.7,\ 0.3) $$ Every row of $A$ is the same vector $r = (0.5, 0.5)$, and the entries of $p_t$ always sum to one. Therefore $$ p_tA = p_1 r + p_2 r = (p_1 + p_2)\, r = r, $$ so the result does not depend on $p_t$: the next state is independent of the current state. 

- $p_tA$ is the distribution over the **next hidden state**: coin $c_1$ and $c_2$ are each used with probability 0.5. 
- $p_tAB$ is the distribution over the **next observation**: heads with probability 0.7 and tails with probability 0.3.

### Q3：

The simplification follows from the observation-independence assumption of the HMM: *given* the current hidden state $X_t = i$, the observation $o_t$ is conditionally independent of all earlier observations. Earlier observations only matter through what they tell us about the state; once the state is fixed, they add no information.

We must still sum over every predecessor $j$ because the state is hidden. The event "observe $o_{1:t}$ and be in state $i$" can happen through any predecessor.

### Q4：

In `forward()`, $\alpha_t(i)$ *sums* over all paths that produce $o_{1:t}$ and end in state $i$; in `viterbi()`, $\delta_t(i)$ keeps only the *best* such path. So Forward outputs one number, the likelihood $P(o_{1:T}\mid\lambda)$, while Viterbi outputs a *state sequence*. Because $\delta$ stores no origin, and the best final path is unknown until time $T$, `psi[t][i]` records every state's best predecessor; we backtrack from the best final state to recover the path.

### Q5：

$\beta_T(i) = 1$ because after time $T$ there are no observations left; the probability of observing an empty sequence is one. 

$\gamma_t(i,j)$ is the probability, given the whole sequence, of being in state $i$ at $t$ and $j$ at $t+1$; $\gamma_t(i) = \sum_j \gamma_t(i,j)$ is the probability of being in $i$ at $t$. Summed over $t$, they give *expected counts* of transitions and state visits. Re-estimating $A$ needs transition counts ($\gamma_t(i,j)$); $B$ and $\pi$ need only state counts ($\gamma_t(i)$).

### Q6：

Convergence rule: In `baum_welch()`, I stop when the log-likelihood $\log P(O\mid\lambda)$ improves by less than $10^{-6}$ over the previous iteration, or after 500 iterations. 

Underflow: In `forward_scaled()`, each $\alpha_t$ is normalised to sum to one, and the factor $c_t = 1/\sum_i \alpha_t(i)$ is stored. `backward_scaled()` scales $\beta_t$ with the same $c_t$. 

Effect on the likelihood: $P(O\mid\lambda)$ itself would underflow, so it is never computed; instead $\log P(O\mid\lambda) = -\sum_t \log c_t$. Because $\alpha$ and $\beta$ share the same factors, their product already contains $1/P$, so $\gamma$ needs no division. A test against the unscaled formulas agreed to $10^{-16}$.

## J3

### 1. Sequence length and convergence

|                                             | N = 1000  | N = 10000 |
| ------------------------------------------- | --------- | --------- |
| Iterations to convergence                   | 750       | 1567      |
| Final log P / T (learned model)             | −1.33701  | −1.34108  |
| log P / T of generating model               | −1.34293  | −1.34187  |
| Learned − generating                        | +0.00592  | +0.00079  |
| Parameter error Σ\|learned − true\| (A + B) | 0.604     | 0.607     |
| Learned π                                   | (1, 0, 0) | (0, 1, 0) |

[N1000] logP/T at iter 1,10,100,500,last: -1.39851, -1.34622, -1.33720, -1.33701, -1.33701

[N10000] logP/T at iter 1,10,100,500,last: -1.38852, -1.34518, -1.34114, -1.34108, -1.34108

With the same initialisation (A₀, B₀, π₀) and stopping rule, training took 750 iterations on N = 1000 and 1567 on N = 10000. 

Part of this difference comes from the stopping rule itself: the tolerance applies to the *total* log-likelihood, so per observation it is ten times stricter for the longer sequence. 

On N = 1000 the learned model scored higher than the generating model (−1.33701 vs −1.34293 per observation), it may fit sampling noise. On N = 10000 this gap shrank from 0.006 to 0.0008. The total parameter error was almost identical (0.604 vs 0.607). 

π converged to a one-hot vector in both cases, since one sequence provides only one sample of the initial state. 

### 2. Initialization

|                         | Uniform                                                      | Exact zeros (diagonal)                                       | Near-true, non-symmetric                                     |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **A₀**                  | all entries 1/3                                              | I (identity)                                                 | (0.60, 0.10, 0.30; 0.15, 0.70, 0.15; 0.25, 0.25, 0.50)       |
| **B₀**                  | all entries 1/4                                              | B₀ from Exp. 1                                               | (0.60, 0.25, 0.10, 0.05; 0.10, 0.35, 0.35, 0.20; 0.05, 0.10, 0.25, 0.60) |
| **π₀**                  | (1/3, 1/3, 1/3)                                              | (0, 0, 1)                                                    | (0.90, 0.05, 0.05)                                           |
| **Iterations**          | 2                                                            | 2                                                            | 1785                                                         |
| **log P / T**           | −1.38137                                                     | −1.38137                                                     | −1.34108                                                     |
| **Σ\|learned − true\|** | 4.200                                                        | 3.665                                                        | 0.604                                                        |
| **Learned model**       | All three states identical; every row of B = symbol frequencies (0.264, 0.270, 0.208, 0.257) | A stays I, π stays (0, 0, 1); only state 3 is used, its B row = symbol frequencies | Same solution as the A₀/B₀/π₀ start of Exp. 1, to three decimals |

Reference (N = 10000): generating model −1.34187; independent symbol-frequency model −1.38137. All runs trained on N = 10000 with the same stopping rule (improvement < 10⁻⁶ or 3000 iterations).

**Uniform:** With the uniform initialisation, all three states have exactly the same parameters, so every state receives the same α and β, and therefore the same γ. As a result, γ_t(i) = 1/3 for every state at every time step, and after the first update every row of B becomes identical: each entry equals the frequency of that observation symbol in the sequence (0.264, 0.270, 0.208, 0.257). A and π remain uniform. In the second iteration all states therefore still have identical parameters, and the updated A, B and π no longer change. The log-likelihood stops improving, so the stopping condition is met and training ends after only two iterations.

**Exact zeros (diagonal):** With A = I and π = (0, 0, 1), the chain starts in state 3 and can never leave it, so the scaled α is (0, 0, 1) at every time step and γt(3) = 1 for all t, while γ is zero for states 1 and 2. Since γt(i, j) ∝ a_ij, a parameter that is exactly zero stays zero, and likewise πi = 0 implies π'i = 0. EM can only redistribute probability among non-zero entries. States 1 and 2 never receive any posterior weight, so their rows keep the initial values, and B's third row stays the same after the first update (2 iterations). The result is again a one-state model, identical in likelihood to the uniform case (−1.38137). 

**Non-symmetric model:** This start converged to the same solution as the A₀/B₀/π₀ start of Experiment 1 (log P/T = −1.34108, parameters equal to three decimals). Two different starting points reaching the same result.

### 3. Number of hidden states and evaluation

I trained models with N = 2, 3 and 4 hidden states on N10000 and evaluated them on N1000 without further updates. All model sizes used the same initialisation procedure (near-uniform random matrices with ±30 % noise, seed = 1000·N + r for restart r) and the same stopping rule, with three restarts per N; the restart with the highest training log-likelihood was kept. 

|                             | N = 2           | N = 3              | N = 4                                   | Generating model |
| --------------------------- | --------------- | ------------------ | --------------------------------------- | ---------------- |
| **Free parameters**         | 9               | 17                 | 27                                      | –                |
| **Train log P/T (N10000)**  | −1.34627        | −1.34108           | **−1.34024**                            | −1.34187         |
| **Eval log P/T (N1000)**    | −1.35249        | **−1.34771**       | −1.37630                                | −1.34293         |
| **Iterations (3 restarts)** | 495 / 350 / 359 | 1781 / 2196 / 1908 | 3000 / 3000 / 3000 (cap)                | –                |
| **Restarts agree?**         | Yes, identical  | Yes, identical     | No: eval −1.35014 / −1.35199 / −1.37630 | –                |

Setup: near-uniform random initialisation (each entry (1/n)(1 + u), u ~ U(−0.3, 0.3), rows normalised), seed = 1000·N + r for restart r ∈ {0, 1, 2}; same stopping rule for all N (improvement < 10⁻⁶ or 3000 iterations); the restart with the highest training log-likelihood was retained.

As N increased, the training score improved steadily (−1.34627, −1.34108, −1.34024), but the evaluation score was best at N = 3 (−1.34771), and the best 4-state model scored worst on the evaluation sequence (−1.37630). More hidden states are therefore not always better: too many states can overfit the training data. For N = 2 and N = 3, all three restarts converged to the same result, whereas for N = 4 all three restarts reached different evaluation scores.

## J4

```
N_HIDDEN = 2                       # 每个物种模型的隐藏状态数
START_STEP = N_STEPS - N_FISH      # 第 110 步开始猜
SEED = 0                           # 随机种子，保证结果可复现
```

