import os.path
import numpy as np
cimport numpy as np
cimport cpython
cimport cython
cimport libc.stdio
cimport libc.stdlib

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
    
    ctypedef struct ograd:
        ograd *next
        int varno
        double coef
        
    ctypedef struct Edaginfo:
        int n_var_
        int nbv_
        int niv_
        int n_con_
        int n_obj_
        int nlo_
        int nranges_
        int nlc_
        int nlnc_
        int nlvb_
        int nlvbi_
        int nlvc_
        int nlvci_
        int nlvo_
        int nlvoi_
        int lnc_
        int nzc_
        int nzo_
        int maxrownamelen_
        int maxcolnamelen_
        int n_conjac_[2]
        int want_xpi0_
        char* objtype_
        ograd** Ograd_
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
    int ampl_sphsetup(ASL*, int, int, int, int)
    double ampl_objval(ASL*, int, double*, int*)
    void ampl_objgrd(ASL*, int, double*, double*, int*)
    void ampl_conval(ASL*, double*, double*, int*)
    int ampl_conival(ASL*, int, double*, double*)

cdef np.ndarray[np.double_t] carray_to_numpy(double *x, int lenx):
    """Utility to copy C array of doubles to numpy array."""
    cdef np.ndarray[np.double_t] \
         v = np.empty(lenx, dtype=np.double)
    cdef int i
    for i in range(lenx):
        v[i] = x[i]
    return v

cdef class ampl:

    cdef ASL* asl
    cdef libc.stdio.FILE* ampl_file

    # Components from Table 1.
    cdef public int n_var
    cdef public int nbv
    cdef public int niv
    cdef public int n_con
    cdef public int n_obj
    cdef public int nlo
    cdef public int nranges
    cdef public int nlc
    cdef public int nlnc
    cdef public int nlvb
    cdef public int nlvbi
    cdef public int nlvc
    cdef public int nlvci
    cdef public int nlvo
    cdef public int nlvoi
    cdef public int lnc
    cdef public int nzc
    cdef public int nzo
    cdef public int maxrownamelen
    cdef public int maxcolnamelen

    cdef public int objtype

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

        # Open stub and get problem dimensions (Table 1 of "Hooking...").
        self.ampl_file = jac0dim_ASL(asl, stub, len(stub))
        
        self.n_var = asl.i.n_var_
        self.nbv = asl.i.nbv_
        self.niv = asl.i.niv_
        self.n_con = asl.i.n_con_
        self.n_obj = asl.i.n_obj_
        self.nlo = asl.i.nlo_
        self.nranges = asl.i.nranges_
        self.nlc = asl.i.nlc_
        self.nlnc = asl.i.nlnc_
        self.nlvb = asl.i.nlvb_
        self.nlvbi = asl.i.nlvbi_
        self.nlvc = asl.i.nlvc_
        self.nlvci = asl.i.nlvci_
        self.nlvo = asl.i.nlvo_
        self.nlvoi = asl.i.nlvoi_
        self.lnc = asl.i.lnc_
        self.nzc = asl.i.nzc_
        self.nzo = asl.i.nzo_
        self.maxrownamelen = asl.i.maxrownamelen_
        self.maxcolnamelen = asl.i.maxcolnamelen_

        # Ask for initial x and pi, and allocate storage for problem data.
        asl.i.want_xpi0_ = 3
        asl.i.X0_    = <double *>malloc(self.n_var * sizeof(double))
        asl.i.LUv_   = <double *>malloc(self.n_var * sizeof(double))
        asl.i.Uvx_   = <double *>malloc(self.n_var * sizeof(double))
        asl.i.pi0_   = <double *>malloc(self.n_con * sizeof(double))
        asl.i.LUrhs_ = <double *>malloc(self.n_con * sizeof(double))
        asl.i.Urhsx_ = <double *>malloc(self.n_con * sizeof(double))

        # Read in the problem.
        pfgh_read_ASL(asl, self.ampl_file, 0)

        self.objtype = asl.i.objtype_[0]

    def get_x0(self): return carray_to_numpy(self.asl.i.X0_, self.n_var) 
    def get_Lvar(self): return carray_to_numpy(self.asl.i.LUv_, self.n_var)
    def get_Uvar(self): return carray_to_numpy(self.asl.i.Uvx_, self.n_var)
    def get_pi0(self): return carray_to_numpy(self.asl.i.pi0_, self.n_con)
    def get_Lcon(self): return carray_to_numpy(self.asl.i.LUrhs_, self.n_con)
    def get_Ucon(self): return carray_to_numpy(self.asl.i.Urhsx_, self.n_con)

    def get_nnzj(self): return self.nzc
    def get_nnzh(self): return ampl_sphsetup(self.asl, -1, 1, 1, 1)

    def get_CType(self):
        nln = range(self.nlc)
        net = range(self.nlc,  self.nlnc)
        lin = range(self.nlc + self.nlnc, self.n_con)
        return (lin, nln, net)

    def eval_obj(self, np.ndarray[np.double_t] x):
        cdef int nerror
        cdef double \
             val = ampl_objval(self.asl, 0, <double*>x.data, &nerror)
        if nerror:
            raise ValueError
        return val

    cpdef grad_obj(self, np.ndarray[np.double_t] x):
        """Evaluate the gradient of the objective at x."""
        cdef int nerror
        cdef np.ndarray[np.double_t] g = x.copy() # slightly faster?
        ampl_objgrd(self.asl, 0, <double*>x.data, <double*>g.data, &nerror)
        if nerror:
            raise ValueError
        return g

    def eval_cons(self, np.ndarray[np.double_t] x):
        """Evaluate the constraints at x."""
        cdef int nerror
        cdef np.ndarray[np.double_t] \
             c = np.empty(self.n_con, dtype=np.double)
        ampl_conval(self.asl, <double*>x.data, <double*>c.data, &nerror)
        if nerror:
            raise ValueError
        return c

    def eval_sgrad(self, np.ndarray[np.double_t] x):
        """Evaluate sparse (linear-part of the) objective gradient at
        the point x passed as argument.  The point x is given in the
        form of an array.  The sparse gradient is returned as a
        dictionary."""
        grad_f = self.grad_obj(x)
        sg = {} ; j = 0
        cdef ograd* og = self.asl.i.Ograd_[0]
        while og is not NULL:
            key = og.varno
            val = grad_f[j]
            sg[key] = val
            og = og.next
            j += 1
        return sg

    def eval_cost(self):
        """Evaluate sparse linear-cost vector."""
        sg = {}
        cdef ograd* og = self.asl.i.Ograd_[0]
        while og is not NULL:
            key = og.varno
            val = og.coef
            sg[key] = val
            og = og.next
        return sg

    def eval_ci(self, int i, np.ndarray[np.double_t] x):
        """Evaluate ith constraint."""
        cdef double ci_of_x
        if i < 0 or i >= self.n_con:
            raise ValueError('Got i = %d; exected 0 <= i < %d' %
                             (i, self.n_con))
        if ampl_conival(self.asl, i, &ci_of_x, <double*>x.data):
            raise ValueError
        return ci_of_x

    def eval_gi(self, i, x):
        pass

    def eval_sgi(self, i, x):
        pass

    def eval_irow(self, i):
        pass

    def eval_A(self):
        pass

    def eval_J(self, x):
        pass

    def eval_H(self, x, mformat, obj_weight, store_zeros):
        pass

    def H_prod(self, z, v, obj_weight):
        pass

    def gHi_prod(self, g, v):
        pass

    def is_lp(self):
        pass

    def set_x(self, x):
        pass

    def unset_x(self):
        pass
    
    def ampl_sol(self, x, z, msg):
        pass
