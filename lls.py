import numpy as np
import matplotlib.pyplot as plt


def fit(x, y):
    """
    Compute least squares solution for linear transformation
    of two devices
    """
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return (m, c)


def fit_multiple(A, B):
    """
    Fit the transformation of two devices for multiple aps or two aps for one device
    where A and B are 2D matrices of shape (n_aps, n_mean_rss)
    """
    coeffs = []
    for i, _ in enumerate(A):
        m, c = fit(A[i], B[i])
        coeffs.append((m, c))
    return coeffs


def transform(x, m, c):
    return x*m + c


def transform_all(A, B, coeffs):
    B_to_A = np.empty(A.shape)
    for i, _ in enumerate(A):
        B_to_A[i] = transform(B[i], coeffs[i][0], coeffs[i][1])
    return B_to_A


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def main():
    x = np.array([-61, -59, -59, -58, -57, -53, -
                  53, -53, -50, -50, -47, -39, -19])
    y = np.array([-100, -96, -95, -94, -93, -92, -
                  91, -90, -90, -89, -87, -83, -78])
    m, c = fit(x, y)
    z = transform(x, m, c)
    x = sorted(x, reverse=True)
    y = sorted(y, reverse=True)
    z = sorted(z, reverse=True)
    plt.plot(x, range(0, 13), color='r', label='x')
    plt.plot(y, range(0, 13), color='b', label='y')
    plt.plot(z, range(0, 13), color='g', label='z')
    plt.title('Linear transformation using m=%5.3f, c=%5.3f' % (m, c))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
