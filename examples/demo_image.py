# A simple image denoising example
# The example 1D image is one of the examples from
#  C. R. Vogel, Computational Methods for Inverse Problems,
#  Frontiers in Applied Mathematics Series #23, SIAM, Philadelphia, 2002.

try:
    from scipy.linalg import toeplitz
except:
    raise ImportError, 'SciPy is required for this demo.'

from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.optimize.solvers import LSQRFramework
from nlpy.krylov.minres import Minres
from math import sqrt
import numpy as np

class Image1D:

    def __init__(self, n=80, sig=.05, err=2, **kwargs):
        self.n = n                 # Number of grid points
        self.sig = float(sig)      # Gaussian kernel width
        self.err = float(err)/100  # Percent error in data

        # Setup grid
        h = 1.0/n
        z = np.arange(h/2, 1-h/2+h, h)

        # Compute nxn matrix K = convolution with Gaussian kernel
        gaussKernel = 1/sqrt(np.pi)/self.sig * np.exp(-(z-h/2)**2/self.sig**2)
        self.K = h * np.matrix(toeplitz(gaussKernel))

        # Setup true solution, blurred and noisy data
        trueimg  = .75 * np.where((.1 < z) & (z < .25), 1, 0)
        trueimg += .25 * np.where((.3 < z) & (z < .32), 1, 0)
        trueimg += np.where((.5 < z) & (z < 1), 1, 0) * np.sin(2*np.pi*z)**4
        blurred = self.K * np.asmatrix(trueimg).T
        blurred = np.asarray(blurred.T)[0]      # np.matrix messes up your data
        noise = self.err * np.linalg.norm(blurred) * np.random.random(n)/sqrt(n)
        self.data = blurred + noise
        self.z = z
        self.trueimg = trueimg
        self.blurred = blurred
        self.setsolver()

    def setsolver(self):
        n = self.n
        op = SimpleLinearOperator(n, n,
                                  lambda u: np.asarray(u * self.K)[0],
                                  matvec_transp=lambda u: np.asarray(u * self.K.T)[0],
                                  symmetric=False)
        self.solver = LSQRFramework(op)

    def deblur(self, **kwargs):
        "Deblur image with specified solver"
        self.solver.solve(self.data, **kwargs)
        return self.solver.x


class Image1DMinres(Image1D):

    def __init__(self, n=80, sig=.05, err=2, **kwargs):
        Image1D.__init__(self, n, sig, err, **kwargs)

    def setsolver(self):
        n = self.n
        op = SimpleLinearOperator(n, n, lambda u: u * self.K, symmetric=True)
        self.solver = Minres(op, check=True, show=True, shift=9.94334578e-01)


class Image1DMinresAug(Image1D):

    def __init__(self, n=80, sig=.05, err=2, **kwargs):
        Image1D.__init__(self, n, sig, err, **kwargs)
        self.reg = kwargs.get('reg',1.0e-3)

    def setsolver(self):
        n = self.n
        op = SimpleLinearOperator(n, n, self.matvec, symmetric=True)
        self.solver = Minres(op, check=True, show=True)

    def matvec(self, x):
        "y <- Ax"
        (m,n) = self.K.shape ; y = np.empty(n+m)
        y[:m] = x[:m] + np.asarray(x[m:] * self.K)[0]
        y[m:] = np.asarray(x[:m] * self.K.T)[0] - self.reg * x[m:]
        return y

    def deblur(self, **kwargs):
        "Deblur image with specified solver"
        b = np.zeros(2*self.n)
        b[:self.n] = self.data
        self.solver.solve(b, **kwargs)
        return self.solver.x


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    damp = 0.1

    img = Image1D(err=1)
    #recon = img.deblur(damp=0.0, show=True)
    recon_damp = img.deblur(damp=damp, show=True)

    #err = np.linalg.norm(img.trueimg-recon)/np.linalg.norm(img.trueimg)
    err2 = np.linalg.norm(img.trueimg-recon_damp)/np.linalg.norm(img.trueimg)
    #print 'Direct error (no damping) = ', err
    print 'Direct error (with damping) = ', err2

    img2 = Image1DMinresAug(err=2, reg=damp**2)
    reconMinres = img2.deblur(itnlim=1000)

    err3 = np.linalg.norm(img.trueimg-reconMinres[img.n:])/np.linalg.norm(img.trueimg)
    print 'Direct error (with damping) = ', err3

    # Plot
    orig = plt.subplot(131)
    orig.plot(img.z, img.trueimg, 'b-', linewidth=1, label='True Image')
    orig.plot(img.z, img.data, 'k.', label='Noisy Data')
    orig.plot(img.z, img.blurred, 'm-', label='Blurred Image')
    orig.set_title('Data')
    orig.legend(loc='lower right')
    orig.set_ylim(-0.2,1)

    lsqr = plt.subplot(132)
    lsqr.plot(img.z, img.trueimg, 'b-', label='True Image')
    #lsqr.plot(img.z, recon, 'g-', label='Deblurred, no Damping')
    lsqr.plot(img.z, recon_damp, 'r-', label='Deblurred, with Damping')
    lsqr.set_title('LSQR')
    lsqr.legend(loc='lower right')
    lsqr.set_ylim(-0.2,1)

    minres = plt.subplot(133)
    minres.plot(img.z, img.trueimg, 'b-', label='True Image')
    minres.plot(img.z, reconMinres[img.n:], 'r-', label='Deblurred')
    minres.set_title('MINRES')
    minres.legend(loc='lower right')
    minres.set_ylim(-0.2,1)
    #plt.savefig('deblur.pdf', bbox_inches='tight')

    plt.show()
