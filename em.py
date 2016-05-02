"""
- Sigma initialization with I * x, x > 1
- Quality depends on the selected box
"""

from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def getbox(x1, y1, x2, y2, image):
    return image[y1:y2, x1:x2]


def both():
    # Number of Gaussians
    K = 5

    # Dimension of data points
    D = 3

    # Read image and get box
    img = misc.imread("banana.png")
    img = np.array(img, dtype=np.float64)

    print("Image shape:")
    print(img.shape)

    # Foreground =====================================
    box = getbox(200, 285, 300, 330, img)
    # box = getbox(200, 285, 480, 365, img)

    mu_f, sigma_f, mix_f = em_try(box, K, D, 10)

    # Background =====================================
    #box = getbox(250, 35, 350, 100, img)
    box = getbox(50, 50, 100, 100, img)

    mu_b, sigma_b, mix_b = em_try(box, K, D, 10)

    ysize, xsize, _ = img.shape

    # xv, yv = np.meshgrid(range(ysize), range(xsize))
    # pixelCoords = np.dstack((xv, yv))
    # pixelCoords = pixelCoords.reshape(-1, 2)

    # numPixel = self.xsize * self.ysize
    # X = np.zeros(
    #     numPixel, dtype=[('y', 'uint16'), ('x', 'uint16'), ('values', 'float64', (numberOfSamples,))])
    # X['y'] = pixelCoords[:, 0]
    # X['x'] = pixelCoords[:, 1]

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

    pass


def test(img, K):
    ysize, xsize, _ = img.shape

    xv, yv = np.meshgrid(range(ysize), range(xsize))
    pixelCoords = np.dstack((xv, yv))
    pixelCoords = pixelCoords.reshape(-1, 2)

    numPixel = xsize * ysize
    X = np.zeros(numPixel, dtype=[('y', 'uint16'), ('x', 'uint16'), ('prob', 'float64', (K,))])
    X['y'] = pixelCoords[:, 0]
    X['x'] = pixelCoords[:, 1]
    pass


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
    print("Box shape:")
    print(box.shape)

    # plt.imshow(box)
    # plt.show()

    # Get data points, shape: (N, D)
    X = box.reshape((box.shape[0] * box.shape[1], D))
    N = X.shape[0]

    print("Data points matrix shape:")
    print(X.shape)

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

    print("Gaussians:")
    print(K)

    print("Data point dimension:")
    print(D)

    print("Initilized parameter:")
    print(mu)
    print(sigma)
    print(mix)

    for it in range(iterations):

        # E-Step

        # Responsibilities
        resp = np.empty((K, N))
        for k in range(K):
            resp[k, :] = mix[k] * scipy.stats.multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])
            # for n in range(N):
            #    resp[k, n] = mix[k] * scipy.stats.multivariate_normal.pdf(X[n], mean=mu[k], cov=sigma[k])
        resp /= resp.sum(0)

        # M-Step

        # mu
        mu_new = np.empty((K, D))
        for k in range(K):
            for n in range(N):
                mu_new[k] += resp[k, n] * X[n]
            mu_new[k] = mu_new[k, :] / resp[k].sum(0)

        # sigma
        sigma_new = np.empty((K, D, D))
        for k in range(K):
            for n in range(N):
                diff = X[n] - mu_new[k]
                sigma_new[k] += resp[k, n] * np.outer(diff, diff)
            sigma_new[k] = sigma_new[k, :] / resp[k].sum(0)

        # mix
        mix_new = np.empty(K)
        for k in range(K):
            Nk = resp[k].sum(0)
            mix_new[k] = Nk / N

        mu = mu_new
        sigma = sigma_new
        mix = mix_new

        # Likelihood
        lik = 0
        for n in range(N):
            tmp = 0
            for k in range(K):
                tmp += resp[k, n] * scipy.stats.multivariate_normal.pdf(X[n], mean=mu[k], cov=sigma[k])
            lik += np.log(tmp)

        print("Iteration " + str(it) + " with likelihood:")
        print(lik)

        pass

    print("Maximized parameter:")
    print("Mu")
    print(mu)
    print("Sigma")
    print(sigma)
    print("Mix")
    print(mix)

    return mu, sigma, mix


def main():
    both()


if __name__ == '__main__':
    main()
