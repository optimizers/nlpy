# A simple image denoising example
# The example 1D image is one of the examples from
#  C. R. Vogel, Computational Methods for Inverse Problems,
#  Frontiers in Applied Mathematics Series #23, SIAM, Philadelphia, 2002.

from nlpy.optimize.solvers import LSQRFramework
from nlpy.krylov.minres import Minres
from scipy.linalg import toeplitz
from math import sqrt
import numpy as np

class Image1D:

    def __init__(self, n=80, sig=.05, err=2, **kwargs):
        self.n = n           # Number of grid points
        self.sig = sig       # Gaussian kernel width
        self.err = err       # Percent error in data

        # Setup grid
        h = 1.0/n
        z = np.arange(h/2, 1-h/2+h, h)

        # Compute nxn matrix K = convolution with Gaussian kernel
        gaussKernel = 1/sqrt(np.pi)/sig * np.exp(-(z-h/2)**2/sig**2)
        self.K = h * np.matrix(toeplitz(gaussKernel))

        # Setup true solution and blurred data
        trueimg = .75 * np.where((.1 < z) & (z < .25), 1, 0) + \
            .25 * np.where((.3 < z) & (z < .32), 1, 0) + \
            np.where((.5 < z) & (z < 1), 1, 0) * np.sin(2*np.pi*z)**4
        blurred = self.K * np.asmatrix(trueimg).T
        blurred = np.asarray(blurred.T)[0]      # np.matrix messes up your data

        noise = err/100 * np.linalg.norm(blurred) * np.random.random(n)/sqrt(n)
        self.data = blurred + noise
        self.z = z
        self.trueimg = trueimg
        self.blurred = blurred
        self.setsolver()

    def setsolver(self):
        n = self.n
        self.solver = LSQRFramework(n, n, self.matvec)

    def matvec(self, mode, m, n, u):
        if mode == 1:
            v = u * self.K
        elif mode == 2:
            v = u * self.K.T
        return np.asarray(v)[0]

    def deblur(self, **kwargs):
        "Deblur image with specified solver"
        self.solver.solve(self.data, **kwargs)
        return self.solver.x


class Image1DMinres(Image1D):

    def __init__(self, n=80, sig=.05, err=2, **kwargs):
        Image1D.__init__(self, n=80, sig=.05, err=2, **kwargs)

    def setsolver(self):
        self.solver = Minres(self.matvec)

    def matvec(self, x, y):
        "y <- Ax"
        (m,n) = self.K.shape
        print 'Shape of input vector x: ', x.shape
        y[:m] = x[:m] + np.asarray(x[m:] * self.K)[0]
        y[m:] = np.asarray(x[:m] * self.K.T)[0]
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    img = Image1D()
    recon = img.deblur(damp=0.0, show=True)
    recon_damp = img.deblur(damp=0.1, show=True)

    err = np.linalg.norm(img.trueimg-recon)/np.linalg.norm(img.trueimg)
    err2 = np.linalg.norm(img.trueimg-recon_damp)/np.linalg.norm(img.trueimg)
    print 'Direct error (no damping) = ', err
    print 'Direct error (with damping) = ', err2

    img2 = Image1DMinres()
    reconMinres = img2.deblur()

    # Plot
    left = plt.subplot(131)
    left.plot(img.z, img.trueimg, 'b-', linewidth=1, label='True Image')
    left.plot(img.z, img.data, 'k.', label='Noisy Data')
    left.plot(img.z, img.blurred, 'm-', label='Blurred Image')
    left.legend(loc='lower right')
    left.set_ylim(-0.2,1)
    #left.set_aspect('equal')

    right = plt.subplot(132)
    right.plot(img.z, img.trueimg, 'b-', label='True Image')
    right.plot(img.z, recon, 'r-', label='Deblurred, no Damping')
    right.plot(img.z, recon_damp, 'g-', label='Deblurred, with Damping')
    right.legend(loc='lower right')
    #right.set_aspect('equal')
    #plt.savefig('deblur.pdf', bbox_inches='tight')

    minres = plt.subplot(133)
    minres.plot(img.z, img.trueimg, 'b-', label='True Image')
    minres.plot(img.z, reconMinres, 'r-', label='Deblurred')
    minres.legend(loc='lower right')

    plt.show()
