import sys


def read_matrix(tokens, pos):
    """读取一个矩阵：行数 列数 元素...，返回 (矩阵, 下一个位置)"""
    rows = int(tokens[pos])
    cols = int(tokens[pos + 1])
    pos += 2
    M = []
    for i in range(rows):
        M.append([float(x) for x in tokens[pos:pos + cols]])
        pos += cols
    return M, pos


def read_sequence(tokens, pos):
    """读取观测序列：长度T o1 o2 ... oT，返回 (序列, 下一个位置)"""
    T = int(tokens[pos])
    pos += 1
    seq = [int(x) for x in tokens[pos:pos + T]]
    pos += T
    return seq, pos


def forward(A, B, pi, obs):
    """
    Forward 算法，返回 P(O_1:T | lambda)
    alpha[i] = P(o_1..o_t, X_t = i)
    """
    N = len(A)

    # 初始化 t = 1: alpha_1(i) = pi_i * b_i(o_1)
    alpha = [pi[i] * B[i][obs[0]] for i in range(N)]

    # 递推 t = 2..T: alpha_t(i) = b_i(o_t) * sum_j alpha_{t-1}(j) * a_ji
    for t in range(1, len(obs)):
        o = obs[t]
        new_alpha = []
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += alpha[j] * A[j][i]
            new_alpha.append(B[i][o] * s)
        alpha = new_alpha

    # 结束：P(O | lambda) = sum_i alpha_T(i)
    return sum(alpha)


def main():
    tokens = sys.stdin.read().split()
    A, pos = read_matrix(tokens, 0)
    B, pos = read_matrix(tokens, pos)
    pi, pos = read_matrix(tokens, pos)
    obs, pos = read_sequence(tokens, pos)

    print(round(forward(A, B, pi[0], obs), 6))


main()
