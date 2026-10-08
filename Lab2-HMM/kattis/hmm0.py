

import sys


def read_matrix(tokens, pos):
    """
    从 tokens 的 pos 位置开始读取一个矩阵。
    格式：行数 列数 元素1 元素2 ...（按行展开）
    返回：(矩阵, 下一个未读位置)
    """
    rows = int(tokens[pos])
    cols = int(tokens[pos + 1])
    pos += 2

    M = []
    for i in range(rows):
        row = [float(x) for x in tokens[pos:pos + cols]]
        M.append(row)
        pos += cols
    return M, pos


def matmul(X, Y):
    """
    矩阵乘法：X 是 n×m，Y 是 m×p，结果是 n×p
    result[i][j] = sum_k X[i][k] * Y[k][j]
    """
    n = len(X)
    m = len(Y)
    p = len(Y[0])

    result = []
    for i in range(n):
        row = []
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += X[i][k] * Y[k][j]
            row.append(s)
        result.append(row)
    return result


def main():
    # 读取全部输入，按空白切成 token 列表
    tokens = sys.stdin.read().split()

    # 依次读取 A (N×N)、B (N×K)、pi (1×N)
    A, pos = read_matrix(tokens, 0)
    B, pos = read_matrix(tokens, pos)
    pi, pos = read_matrix(tokens, pos)

    # 第 1 步：下一时刻的状态分布  p = pi · A    (1×N)
    next_state = matmul(pi, A)

    # 第 2 步：下一时刻的观测分布  q = p · B     (1×K)
    next_obs = matmul(next_state, B)

    # 输出格式：1 K q1 q2 ... qK
    row = next_obs[0]
    print(row)
    print(1, len(row), " ".join(str(round(x, 6)) for x in row))


main()
