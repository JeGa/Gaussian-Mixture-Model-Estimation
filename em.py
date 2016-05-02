from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def getbox(x1, y1, x2, y2, image):
    return image[y1:y2, x1:x2]


def em():
    # Number of Gaussians
    K = 5

    # Dimension of data points
    D = 3

    # Read image and get box
    img = misc.imread("banana.png")
    img = np.array(img, dtype=np.float64)

    print("Image shape:")
    print(img.shape)

    # 480, 365
    box = getbox(200, 285, 240, 300, img)

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

    for it in range(20):

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
            mu_new[k] = mu_new[k,:] / resp[k].sum(0)

        # sigma
        sigma_new = np.empty((K, D, D))
        for k in range(K):
            for n in range(N):
                diff = X[n] - mu_new[k]
                sigma_new[k] += resp[k, n] * np.outer(diff, diff)
            sigma_new[k] = sigma_new[k,:] / resp[k].sum(0)

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

    pix = img[220, 290]

    ysize, xsize, _ = img.shape

    # xv, yv = np.meshgrid(range(ysize), range(xsize))
    # pixelCoords = np.dstack((xv, yv))
    # pixelCoords = pixelCoords.reshape(-1, 2)

    # numPixel = self.xsize * self.ysize
    # X = np.zeros(
    #     numPixel, dtype=[('y', 'uint16'), ('x', 'uint16'), ('values', 'float64', (numberOfSamples,))])
    # X['y'] = pixelCoords[:, 0]
    # X['x'] = pixelCoords[:, 1]

    #for k in range(K)

    for y in range(ysize):
        for x in range(xsize):
            for k in range(K):
                prob = scipy.stats.multivariate_normal.pdf(img[y, x], mean=mu[k], cov=sigma[k])
                if prob >= 1.12140262525e-45:
                    img[y, x] = np.array([255, 255, 0])
                    continue
                else:
                    img[y, x] = np.array([0, 0, 0])

    plt.imshow(img)
    plt.show()

    # print(prob)
    #
    # pix = img[10, 10]
    #
    # for k in range(K):
    #     prob = scipy.stats.multivariate_normal.pdf(pix, mean=mu[k], cov=sigma[k])
    #     print(prob)


def main():
    em()


if __name__ == '__main__':
    main()