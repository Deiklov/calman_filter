import math
import numpy as np
import matplotlib.pyplot as plt


def readExperimentalData(file_name: str) -> np.ndarray:
    with open(file_name) as f:
        lines = f.read().splitlines()
        lines = [x.strip(' ') for x in lines]
        lines = [float(x.strip(',')) for x in lines]
        return np.asarray(lines)


def run_calman():
    Q = pow(10, -4)
    R = 0.06 * 0.06
    J = 4.608 * pow(10, -7)
    T = 0.08
    t = 0.001
    r = 12 * pow(10, -3)
    C1 = r * 10
    #     measured data
    z = readExperimentalData('0001.txt')
    datalen = len(z)
    # part1 = 20
    # transition_matrices
    PHI = np.array([[1, t, 0],
                    [0, 1, t / J],
                    [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    b = np.array([0, 0, 1]).reshape(3, 1)
    # control=C1,
    x = np.zeros((3, datalen + 1), dtype=float)
    # ввели априорную оценку для 1 измерения
    x[:, 0] = [z[0], (z[1] - z[0]) / t, 0.1 * 0.392 * r]

    P = np.zeros((datalen + 1, 3, 3), dtype=float)
    # инициализировали матрицу ковариаций для 1 измерения
    P[0] = [[0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1]]
    for ind, val in enumerate(x.T[:-1]):
        invMatr = 1 / (H.dot(P[ind].dot(H.T)) + R)
        K = P[ind].dot(H.T) * invMatr
        x[:, ind] = x[:, ind] + K.dot(z[ind] - H.dot(x[:, ind]))
        P[ind] = P[ind] - K.dot(H).dot(P[ind])
        #  посчитали априорную для следующего шага
        x[:, ind + 1] = PHI.dot(x[:, ind]) + (C1 * b).reshape(3, )
        P[ind + 1] = PHI.dot(P[ind]).dot(PHI.T) + Q
    print(x[0])

    estimate_arr = x[0][:-1]
    plt.plot(np.arange(t, t * (datalen + 1), t), z[:], label="Z")
    plt.plot(np.arange(t, t * (datalen + 1), t), estimate_arr[:], label="Estimate")
    plt.ylabel('angles')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Dispersion x1,x2,x3')
    ax1.plot(np.arange(t, t * (datalen + 1), t), P[:, 0, 0][:-1])
    ax1.set_title("x1")
    ax2.plot(np.arange(t, t * (datalen + 1), t), P[:, 1, 1][:-1])
    ax2.set_title("x2")
    ax3.plot(np.arange(t, t * (datalen + 1), t), P[:, 2, 2][:-1])
    ax3.set_title("x3")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_calman()
