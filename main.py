import math

import numpy as np

import matplotlib.pyplot as plt

from pykalman import KalmanFilter


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


def readExperimentalData(file_name: str) -> np.ndarray:
    with open(file_name) as f:
        lines = f.read().splitlines()
        lines = [x.strip(' ') for x in lines]
        lines = [float(x.strip(',')) for x in lines]
        return np.asarray(lines)


def run_calman(name):
    Q = pow(10, -1)
    R = 0.06 * 0.06
    J = 4.608 * pow(10, -7)
    T = 0.08
    t = 0.001
    r = 12 * pow(10, -3)
    C1 = r * 10
    #     measured data
    z = readExperimentalData('0001.txt')
    datalen = len(z)
    part1 = 20
    # transition_matrices
    PHI = np.array([[1, t, 0],
                    [0, 1, t / J],
                    [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    b = np.array([0, 0, 1]).reshape(3, 1)
    # control=C1,
    x = np.zeros((3, part1 + 1), dtype=float)
    # ввели априорную оценку для 1 измерения
    x[:, 0] = [z[0], (z[1] - z[0]) / t, 0.1 * 0.392 * r]

    P = np.zeros((part1 + 1, 3, 3), dtype=float)
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

    x2 = np.zeros((3, datalen + 1 - part1), dtype=float)
    # ввели априорную оценку для 1 измерения
    x2[:, 0] = x[:, part1]

    P2 = np.zeros((datalen + 1 - part1, 3, 3), dtype=float)
    # инициализировали матрицу ковариаций для 1 измерения
    P2[0] = P[part1]

    for ind, val in enumerate(x2.T[:-1]):
        invMatr = 1 / (H.dot(P2[ind].dot(H.T)) + R)
        K = P2[ind].dot(H.T) * invMatr
        x2[:, ind] = x2[:, ind] + K.dot(z[ind] - H.dot(x2[:, ind]))

        P2[ind] = P2[ind] - K.dot(H).dot(P2[ind])
        #  посчитали априорную для следующего шага
        x2[:, ind + 1] = PHI.dot(x2[:, ind]) + (-C1 * b).reshape(3, )
        P2[ind + 1] = PHI.dot(P2[ind]).dot(PHI.T) + Q
    print(x2[0])
    estimate_arr = np.concatenate((x[0][:-1], x2[0][:-1]))
    plt.plot(np.arange(0.001, 0.001 * (datalen + 1), 0.001), z[:], label="Z")
    plt.plot(np.arange(0.001, 0.001 * (datalen + 1), 0.001), estimate_arr[:], label="Estimate")
    plt.ylabel('angles')
    plt.xlabel('time')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_calman('PyCharm')
