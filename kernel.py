# import numpy as np
# from sklearn.neighbors.kde import KernelDensity
# import matplotlib.pyplot as plt
# from scipy.stats import norm


# # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# X = np.linspace(-5, 10, 100)[:, np.newaxis]
# true_dens = (0.3 * norm(0, 1).pdf(X[:, 0])
#              + 0.7 * norm(5, 1).pdf(X[:, 0]))
# kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
# print([[x, t] for x in X for t in true_dens])
# y = kde.score_samples(X)
# plt.scatter(X[:, 0], true_dens, color='b')
# plt.scatter(X[:, 0], np.exp(y), color='r')
# # plt.scatter(X[:, 0], X[:, 1], color='b', label='original')
# # plt.scatter(X[:, 0], y, color='r', label='kernel')
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# #----------------------------------------------------------------------
# # Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]
print(X)

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
print(X_plot)

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()
