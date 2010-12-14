import os.path
import numpy as np
cimport numpy as np
cimport cpython
cimport libc.stdio
cimport libc.stdlib

#@cython.boundscheck(False)
cdef enum:
    ASL_read_f    = 1
    ASL_read_fg   = 2
    ASL_read_fgh  = 3
    ASL_read_pfg  = 4
    ASL_read_pfgh = 5

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)

cdef extern from "amplutils.h":
    
    ctypedef struct Edaginfo:
        int n_var_
        int n_con_
        int nlc_
        int nlnc_
        int want_xpi0_
        char* objtype_
        double* X0_
        double* pi0_
        double* LUv_
        double* Uvx_
        double* LUrhs_
        double* Urhsx_
    ctypedef struct ASL:
        Edaginfo i

    ASL* ASL_alloc(int)
    void* ASL_free(ASL**)
    libc.stdio.FILE* jac0dim_ASL(ASL*, char*, int)
    int pfgh_read_ASL(ASL*, libc.stdio.FILE*, int)
    int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b)

cdef np.ndarray[np.double_t, ndim=1] carray_to_numpy(double *x, int lenx):
    """Utility to copy C array of doubles to numpy array."""
    cdef np.ndarray[np.double_t, ndim=1] \
         v = np.empty(lenx, dtype=np.double)
    cdef int i
    for i in range(lenx):
        v[i] = x[i]
    return v

cdef class ampl:

    cdef ASL* asl
    cdef libc.stdio.FILE* ampl_file
    cdef public int _objtype
    cdef public int _n_var, _n_con, _nlc, _nlnc

    def __cinit__(self):
        """cinit is called before init; allocates the ASL structure."""
        self.asl = ASL_alloc(ASL_read_pfgh)
        if self.asl is NULL:
            cpython.PyErr_NoMemory()

    def __dealloc__(self):
        """Free the allocated memory and ASL structure."""
        free(self.asl.i.X0_)
        free(self.asl.i.LUv_)
        free(self.asl.i.Uvx_)
        free(self.asl.i.pi0_)
        free(self.asl.i.LUrhs_)
        free(self.asl.i.Urhsx_)
        if self.asl is not NULL:
            ASL_free(&self.asl)

    def __init__(self, stub):
        """Initialize an ampl object."""
        asl = self.asl

        # Let Python try to open the file before giving it to
        # Ampl. Any exception should be caught by the caller.
        basename, extension = os.path.splitext(stub)
        if len(extension) == 0:
            stub += '.nl' # add the nl extension
        f = open(stub,'r'); f.close()

        # Open stub and get problem dimensions.
        self.ampl_file = jac0dim_ASL(asl, stub, len(stub))
        self._n_var = asl.i.n_var_
        self._n_con = asl.i.n_con_
        self._nlc = asl.i.nlc_
        self._nlnc = asl.i.nlnc_

        # Ask for initial x and pi, and allocate storage for problem data.
        asl.i.want_xpi0_ = 3
        asl.i.X0_    = <double *>malloc(self._n_var * sizeof(double))
        asl.i.LUv_   = <double *>malloc(self._n_var * sizeof(double))
        asl.i.Uvx_   = <double *>malloc(self._n_var * sizeof(double))
        asl.i.pi0_   = <double *>malloc(self._n_con * sizeof(double))
        asl.i.LUrhs_ = <double *>malloc(self._n_con * sizeof(double))
        asl.i.Urhsx_ = <double *>malloc(self._n_con * sizeof(double))

        # Read in the problem.
        pfgh_read_ASL(asl, self.ampl_file, 0)

        self._objtype = asl.i.objtype_[0]

    def get_x0(self): return carray_to_numpy(self.asl.i.X0_, self._n_var) 
    def get_Lvar(self): return carray_to_numpy(self.asl.i.LUv_, self._n_var)
    def get_Uvar(self): return carray_to_numpy(self.asl.i.Uvx_, self._n_var)
    def get_pi0(self): return carray_to_numpy(self.asl.i.pi0_, self._n_con)
    def get_Lcon(self): return carray_to_numpy(self.asl.i.LUrhs_, self._n_con)
    def get_Ucon(self): return carray_to_numpy(self.asl.i.Urhsx_, self._n_con)

    def get_CType(self):
        nln = range(self._nlc)
        return nln
