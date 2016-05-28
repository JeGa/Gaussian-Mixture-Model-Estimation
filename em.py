"""
- Sigma initialization with I * x, x > 1
- Quality depends on the selected box
"""

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import logging


def getbox(x1, y1, x2, y2, image):
    return image[y1:y2, x1:x2]


def both():
    """
        Do EM for foreground and background bounding box.
    """

    logging.basicConfig(level=logging.ERROR)

    # Number of Gaussians
    K = 5

    # Dimension of data points
    D = 3

    # Read image and get box
    img = misc.imread("banana.png")
    img = np.array(img, dtype=np.float64)

    logging.info("Image shape: " + str(img.shape))

    # Foreground =====================================
    box = getbox(200, 285, 400, 330, img)
    # box = getbox(200, 285, 480, 365, img)
    plt.imshow(box)

    print("Run foreground")
    mu_f, sigma_f, mix_f = em_try(box, K, D, 10)

    # Background =====================================
    # box = getbox(250, 35, 350, 100, img)
    box = getbox(50, 50, 200, 100, img)
    plt.imshow(box)

    print("Run background")
    mu_b, sigma_b, mix_b = em_try(box, K, D, 10)

    ysize, xsize, _ = img.shape

    print("Compare results")

    gauss_f = []
    gauss_b = []
    for k in range(K):
        gauss_f.append(scipy.stats.multivariate_normal(mu_f[k], sigma_f[k]))
        gauss_b.append(scipy.stats.multivariate_normal(mu_b[k], sigma_b[k]))

    for y in range(ysize):
        for x in range(xsize):
            prob_f = 0
            prob_b = 0
            for k in range(K):
                prob_f += mix_f[k] * gauss_f[k].pdf(img[y, x])
                prob_b += mix_f[k] * gauss_b[k].pdf(img[y, x])

            if prob_f > prob_b:
                img[y, x] = np.array([255, 255, 0])
            else:
                img[y, x] = np.array([0, 0, 0])

    img = img.astype(np.uint8)

    plt.imshow(img)
    plt.show()
    np.save("imgdata", img)

    np.savez("prob_foreground", mu_f, sigma_f, mix_f)
    np.savez("prob_background", mu_b, sigma_b, mix_b)


def em_try(box, K, D, iterations):
    while True:
        try:
            mu, sigma, mix = em(box, K, D, iterations)
        except:
            print("Try ...")
            continue
        break
    return mu, sigma, mix


def em(box, K, D, iterations):
    """
    Arguments:
        box: Subset of the image
        K: Number of Gaussians
        D: Dimension of the data points
        iterations: Number of EM iterations
    """

    logging.info("Box shape: " + str(box.shape))
    # plt.imshow(box)
    # plt.show()

    # Get data points, shape: (N, D)
    X = box.reshape((box.shape[0] * box.shape[1], D))
    N = X.shape[0]

    logging.info("Data points matrix shape: " + str(X.shape))

    # Mixture of gaussian parameter (theta)
    mu = np.empty((K, D))
    sigma = np.empty((K, D, D))
    mix = np.empty(K)

    # Initialize mu to random data samples
    index = np.random.randint(0, X.shape[0], K)
    mu = X[index]

    # Identity matrix
    sigma[:] = 3 * np.identity(D)

    # Sum over all mixing coefficients = 1
    mix[:] = 1.0 / K

    logging.info("Gaussians: " + str(K))
    logging.info("Data point dimension: " + str(D))
    logging.info("Initialized parameter:")
    logging.info(mu)
    logging.info(sigma)
    logging.info(mix)

    # Marginal log likelihood p(X|theta)
    lik = 0

    for it in range(iterations):
        # E-Step, get posterior over responsibilities p(Z|X)

        # Responsibilities
        resp = np.empty((K, N))
        for k in range(K):
            resp[k, :] = mix[k] * scipy.stats.multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
        sum = resp.sum(0)
        # Replace all zeors with 1 so we do not divide by 0.
        # The resulting responsibility for that point will be then 0/1 = 0.
        sum[sum == 0] = 1
        resp /= sum

        # M-Step, maximize log likelihood for mu, sigma and mix

        # Precompute
        Nk = resp.sum(1)

        # mu
        mu_new = np.empty((K, D))
        for k in range(K):
            for n in range(N):
                mu_new[k] += resp[k, n] * X[n]
            mu_new[k] = mu_new[k, :] / Nk[k]

        # sigma
        sigma_new = np.empty((K, D, D))
        for k in range(K):
            for n in range(N):
                diff = X[n] - mu_new[k]
                sigma_new[k] += resp[k, n] * np.outer(diff, diff)
            sigma_new[k] = sigma_new[k, :] / Nk[k]

        # mix
        mix_new = np.empty(K)
        for k in range(K):
            mix_new[k] = Nk[k] / N

        mu = mu_new
        sigma = sigma_new
        mix = mix_new

        # Likelihood
        lik_new = 0
        for n in range(N):
            tmp = 0
            for k in range(K):
                tmp += mix[k] * scipy.stats.multivariate_normal.pdf(X[n], mean=mu[k], cov=sigma[k])
            lik_new += np.log(tmp)

        print("Iteration " + str(it) + " with likelihood " + str(lik_new))

        if np.abs(lik - lik_new) <= 0.5:
            break

        lik = lik_new

    logging.info("Maximized parameter:")
    logging.info(mu)
    logging.info(sigma)
    logging.info(mix)

    return mu, sigma, mix


def main():
    both()


if __name__ == '__main__':
    main()
