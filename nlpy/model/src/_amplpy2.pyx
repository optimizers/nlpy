import os.path
from pysparse import spmatrix
cimport cpython
cimport cython
cimport libc.stdio
from libc.stdlib cimport free, malloc

########################################################################
# AMPL headers
########################################################################
cdef enum:
    ASL_read_f    = 1
    ASL_read_fg   = 2
    ASL_read_fgh  = 3
    ASL_read_pfg  = 4
    ASL_read_pfgh = 5

cdef extern from "amplutils.h":

    # Option_Info stuff. Only "wantsol" is currently used.
    ctypedef struct keyword:
        pass
    ctypedef struct Solver_KW_func:
        pass
    ctypedef struct Fileeq_func:
        pass
    ctypedef struct Option_Info:
        char*  sname
        char* bsname
        char* opname
        keyword *keywds
        int n_keywds
        int want_funcadd
        char *version
        char **usage
        Solver_KW_func *kwf
        Fileeq_func *feq
        keyword *options
        int n_options
        int wantsol

    ctypedef struct cgrad:
        cgrad *next
        int varno
        int goff
        double coef

    ctypedef struct ograd:
        ograd *next
        int varno
        double coef

    ctypedef struct SputInfo:
        int *hcolstarts
        int *hrownos
        
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
        int congrd_mode
        char* objtype_
        ograd** Ograd_
        cgrad** Cgrad_
        double* X0_
        double* pi0_
        double* LUv_
        double* Uvx_
        double* LUrhs_
        double* Urhsx_
        int x_known
        SputInfo* sputinfo_

    ctypedef struct ASL:
        Edaginfo i

    ASL* ASL_alloc(int)
    void* ASL_free(ASL**)
    libc.stdio.FILE* jac0dim_ASL(ASL*, char*, int)
    int pfgh_read_ASL(ASL*, libc.stdio.FILE*, int)
    void write_sol_ASL(ASL*, char*, double*, double*, Option_Info*)
    int ampl_sphsetup(ASL*, int, int, int, int)
    double ampl_objval(ASL*, int, double*, int*)
    int ampl_objgrd(ASL*, int, double*, double*)
    int ampl_conval(ASL*, double*, double*)
    int ampl_jacval(ASL*, double*, double*)
    int ampl_conival(ASL*, int, double*, double*)
    int ampl_congrd(ASL*, int, double*, double*)
    void ampl_sphes(ASL*, double*, int, double*, double*)
    void ampl_hvcomp(ASL*, double*, double*, int, double*, double*)
    void ampl_xknown(ASL*, double*)

########################################################################
# PySparse headers
########################################################################
cdef enum:
    SYMMETRIC = 1    # Symmetric SpMatrix
    GENERAL   = 0    # General   SpMatrix
    
cdef extern from "ll_mat.h":
    pass

cdef extern from "spmatrix_api.h":
    void import_spmatrix()
import_spmatrix() # this does the actual import

cdef extern from "spmatrix.h":
    int SpMatrix_LLMatSetItem(void* self, int i, int j, double val)
    object SpMatrix_NewLLMatObject(int* dims, int type, int nnz, int store)

########################################################################
# Numpy utility function
########################################################################
import numpy as np
cimport numpy as np
from numpy cimport npy_intp, NPY_DOUBLE, ndarray

## cdef extern from "Python.h":
##     void Py_INCREF(object)
##     ctypedef struct PyObject:
##         pass

cdef extern from "numpy/arrayobject.h":
    bint PyArray_ISCARRAY(ndarray)
    ndarray PyArray_EMPTY(int, npy_intp*, int, int)
    void import_array()
    ## ndarray PyArray_GETCONTIGUOUS(ndarray)
    ## int PyArray_AsCArray(PyObject**, void*, npy_intp*, int, np.dtype)
    ## np.dtype PyArray_DescrFromType(int)
    
import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ndarray copy_c_to_numpy(double *x, int lenx):
    """Utility to copy C array of doubles to numpy array."""
    cdef:
        npy_intp* dims = [lenx]
        ndarray[np.double_t] \
            v = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0)
        int i
    
    for i in range(lenx):
        v[i] = x[i]
    return v

## cdef double* get_data_ptr(ndarray x, int lenx):
##     """Return a pointer to data guaranteed to be C-contiguous (and
##     correct length). Probably could use PyArray_GETCONGIGOUS here, but
##     it seems that PyArray_ISCARRAY might be faster if it tests true,
##     which is the case we most often expect."""
##     cdef:
##         double* ptr
##         int err
##         npy_intp* dims = [lenx]
##         PyObject *tmp = <PyObject*>x
        
##     if x.shape[0] != lenx:
##         raise ValueError('Input array is incorrect length.')
##     if PyArray_ISCARRAY(x):
##         return <double*>x.data
##     else:
##         err = PyArray_AsCArray(&tmp, <void*>ptr, dims, 1,
##                                PyArray_DescrFromType(NPY_DOUBLE))
##         if err < 0:
##             raise TypeError('Unable to covnert to C array.')
##         else:
##             return ptr

########################################################################
# AMPL interface class
########################################################################
cdef class ampl:

    cdef:
        ASL* asl
        libc.stdio.FILE* ampl_file
        Option_Info Oinfo

        # Components from Table 1.
        public int n_var
        public int nbv
        public int niv
        public int n_con
        public int n_obj
        public int nlo
        public int nranges
        public int nlc
        public int nlnc
        public int nlvb
        public int nlvbi
        public int nlvc
        public int nlvci
        public int nlvo
        public int nlvoi
        public int lnc
        public int nzc
        public int nzo
        public int maxrownamelen
        public int maxcolnamelen
        
        # Other odds and ends.
        public int objtype
        public bint ampl_written_sol

    def __cinit__(self):
        """cinit is called before init; allocates the ASL structure."""

        # Allocate the ASL object.
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
        
        # Let Python try to open the file before giving it to
        # Ampl. Any exception should be caught by the caller.
        basename, extension = os.path.splitext(stub)
        if len(extension) == 0:
            stub += '.nl' # add the nl extension
        f = open(stub,'r'); f.close()

        # Open stub and get problem dimensions (Table 1 of "Hooking...").
        self.ampl_file = jac0dim_ASL(self.asl, stub, len(stub))
        
        self.n_var = self.asl.i.n_var_
        self.nbv = self.asl.i.nbv_
        self.niv = self.asl.i.niv_
        self.n_con = self.asl.i.n_con_
        self.n_obj = self.asl.i.n_obj_
        self.nlo = self.asl.i.nlo_
        self.nranges = self.asl.i.nranges_
        self.nlc = self.asl.i.nlc_
        self.nlnc = self.asl.i.nlnc_
        self.nlvb = self.asl.i.nlvb_
        self.nlvbi = self.asl.i.nlvbi_
        self.nlvc = self.asl.i.nlvc_
        self.nlvci = self.asl.i.nlvci_
        self.nlvo = self.asl.i.nlvo_
        self.nlvoi = self.asl.i.nlvoi_
        self.lnc = self.asl.i.lnc_
        self.nzc = self.asl.i.nzc_
        self.nzo = self.asl.i.nzo_
        self.maxrownamelen = self.asl.i.maxrownamelen_
        self.maxcolnamelen = self.asl.i.maxcolnamelen_

        # Ask for initial x and pi, and allocate storage for problem data.
        self.asl.i.want_xpi0_ = 3
        self.asl.i.X0_    = <double *>malloc(self.n_var * sizeof(double))
        self.asl.i.LUv_   = <double *>malloc(self.n_var * sizeof(double))
        self.asl.i.Uvx_   = <double *>malloc(self.n_var * sizeof(double))
        self.asl.i.pi0_   = <double *>malloc(self.n_con * sizeof(double))
        self.asl.i.LUrhs_ = <double *>malloc(self.n_con * sizeof(double))
        self.asl.i.Urhsx_ = <double *>malloc(self.n_con * sizeof(double))

        # Read in the problem.
        pfgh_read_ASL(self.asl, self.ampl_file, 0)

        # Maximization or minimization.
        self.objtype = self.asl.i.objtype_[0] # 0 = minimization


    # Routines to get initial values.
    def get_x0(self): return copy_c_to_numpy(self.asl.i.X0_, self.n_var) 
    def get_Lvar(self): return copy_c_to_numpy(self.asl.i.LUv_, self.n_var)
    def get_Uvar(self): return copy_c_to_numpy(self.asl.i.Uvx_, self.n_var)
    def get_pi0(self): return copy_c_to_numpy(self.asl.i.pi0_, self.n_con)
    def get_Lcon(self): return copy_c_to_numpy(self.asl.i.LUrhs_, self.n_con)
    def get_Ucon(self): return copy_c_to_numpy(self.asl.i.Urhsx_, self.n_con)

    # Sparsity of Jacobian and Hessian.
    cpdef get_nnzj(self): return self.nzc
    cpdef get_nnzh(self): return ampl_sphsetup(self.asl, -1, 1, 1, 1)

    def get_CType(self):
        nln = range(self.nlc)
        net = range(self.nlc,  self.nlnc)
        lin = range(self.nlc + self.nlnc, self.n_con)
        return (lin, nln, net)

    def eval_obj(self, ndarray[np.double_t] x):
        cdef:
            int nerror
            double val

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        val = ampl_objval(self.asl, 0, <double*>x.data, &nerror)
        if nerror:
            raise ValueError
        return val

    cpdef grad_obj(self, ndarray[np.double_t] x):
        """Evaluate the gradient of the objective at x."""
        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()
        cdef ndarray[np.double_t] g = x.copy()
        if ampl_objgrd(self.asl, 0, <double*>x.data, <double*>g.data):
            raise ValueError
        return g

    def eval_cons(self, ndarray[np.double_t] x):
        """Evaluate the constraints at x."""
        cdef ndarray[np.double_t] \
             c = np.empty(self.n_con, dtype=np.double)

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        if ampl_conval(self.asl, <double*>x.data, <double*>c.data):
            raise ValueError
        return c

    def eval_sgrad(self, ndarray[np.double_t] x):
        """Evaluate linear-part of the objective gradient at x.  A
        sparse gradient is returned as a dictionary."""
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

    def eval_ci(self, int i, ndarray[np.double_t] x):
        """Evaluate ith constraint."""
        cdef double ci
        if i < 0 or i >= self.n_con:
            raise ValueError('Got i = %d; exected 0 <= i < %d' %
                             (i, self.n_con))

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        if ampl_conival(self.asl, i, <double*>x.data, &ci):
            raise ValueError
        return ci

    def eval_gi(self, int i, ndarray[np.double_t] x):
        """Evaluate the ith constraint gradient at x."""
        if i < 0 or i >= self.n_con:
            raise ValueError('Got i = %d; exected 0 <= i < %d' %
                             (i, self.n_con))

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        cdef ndarray[np.double_t] \
             gi = np.empty(self.n_var, dtype=np.double)
        if ampl_congrd(self.asl, i, <double*>x.data, <double*>gi.data):
            raise ValueError
        return gi

    def eval_sgi(self, int i, ndarray[np.double_t] x):
        """Evalute the ith constraint sparse gradient at x."""

        cdef:
            int nzgi, j
            cgrad* cg

        if i < 0 or i >= self.n_con:
            raise ValueError('Got i = %d; exected 0 <= i < %d' %
                             (i, self.n_con))

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        # Set sparse format for gradient. (Restore saved val later.)
        congrd_mode_save = self.asl.i.congrd_mode
        self.asl.i.congrd_mode = 1

        # Count number of nonzeros in gi.
        nzgi = 0
        cg = self.asl.i.Cgrad_[i]
        while cg is not NULL:
            nzgi += 1
            cg = cg.next

        # Allocate storage and evaluate ith constraint at x.
        cdef ndarray[np.double_t] \
             grad_ci = np.empty(nzgi, dtype=np.double)
        if ampl_congrd(self.asl, i, <double*>x.data, <double*>grad_ci.data):
            raise ValueError('congrd failed')
        
        # Generate dictionary.
        j = 0
        sgi = {}
        cg = self.asl.i.Cgrad_[i]
        while cg is not NULL:
            sgi[cg.varno] = grad_ci[j]
            cg = cg.next
            j += 1

        # Restore gradient mode
        self.asl.i.congrd_mode = congrd_mode_save
    
        return sgi

    def eval_row(self, int i):
        """Evaluate the ith constraint gradient as a sparse vector. To
        be used when the problem is a linear program."""
        if i < 0 or i >= self.n_con:
            raise ValueError('Got i = %d; exected 0 <= i < %d' %
                             (i, self.n_con))
        row = {}
        cdef cgrad* cg = self.asl.i.Cgrad_[i]
        while cg is not NULL:
            row[cg.varno] = cg.coef
            cg = cg.next
        return row

    def eval_A(self, int store_zeros=0, spJac=None):
        """Evaluate Jacobian of LP."""
        cdef:
            cgrad* cg
            int *dims = [self.n_con, self.n_var]
            int i, irow, jcol
        nnzj = self.nzc if self.n_con else 1

        # Determine room necessary for Jacobian.
        if spJac is None:
            spJac = SpMatrix_NewLLMatObject(dims, GENERAL,
                                            nnzj, store_zeros)

        # Create sparse Jacobian structure.
        for i in range(self.n_con):
            irow = <long>i
            cg = self.asl.i.Cgrad_[i]
            while cg is not NULL:
                jcol = <long>cg.varno
#               spJac[irow, jcol] = cg.coef
                SpMatrix_LLMatSetItem(<void*>spJac, irow, jcol, cg.coef)
                cg = cg.next

        return spJac

    def eval_J(self, ndarray[np.double_t] x, coord, int store_zeros=0, spJac=None):
        """Evaluate sparse Jacobian."""
        cdef:
            ndarray[np.double_t] J

            # Variables needed for coordinate format.
            ndarray[np.int_t] a_irow, a_icol

            # Variables needed for LL format.
            cgrad* cg
            int* dims = [self.n_con, self.n_var]

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()

        nnzj = self.nzc if self.n_con else 1

        # Allocate storage and evaluate Jacobian at x.
        J = np.empty(nnzj, dtype=np.double)
        if ampl_jacval(self.asl, <double*>x.data, <double*>J.data):
            raise ValueError

        if coord: # return Jacobian in coordinate format.
            a_icol = np.empty(nnzj, dtype=np.int)
            a_irow = np.empty(nnzj, dtype=np.int)

            for i in range(self.n_con):
                cg = self.asl.i.Cgrad_[i]
                while cg is not NULL:
                    a_irow.data[cg.goff] = i
                    a_icol.data[cg.goff] = cg.varno
                    cg = cg.next

            # Return the triple (J, irow, icol).
            return (J, a_irow, a_icol)

        else: # return Jacobian in LL format.

            # Determine room necessary for Jacobian.
            if spJac is None:            
                spJac = SpMatrix_NewLLMatObject(dims, GENERAL,
                                                nnzj, store_zeros)
                        
            # Create sparse Jacobian structure.
            for i in range(self.n_con):
                cg = self.asl.i.Cgrad_[i]
                while cg is not NULL:
                    irow = i
                    jcol = cg.varno
                    SpMatrix_LLMatSetItem(<void*>spJac, irow, jcol, J[cg.goff])
                    cg = cg.next
                    
            return spJac
        
    def eval_H(self, ndarray[np.double_t] x, ndarray[np.double_t] y,
               coord, double obj_weight=1.0, int store_zeros=0, spHess=None):
        """Evaluate sparse upper triangle of Lagrangian Hessian.
        
        NOTE: .... why are we passing x ???
        
        In the future, we will want to be careful here, in case x has
        changed but f(x), c(x) or J(x) have not yet been recomputed. In
        such a case, Ampl has NOT updated the data structure for the
        Hessian, and it will still hold the Hessian at the last point at
        which, f, c or J were evaluated !"""
        cdef:
            ndarray[np.double_t] H

            # Variables needed for coordinate format.
            ndarray[np.int_t] a_irow, a_icol

            # Variables needed for LL format.
            cgrad* cg
            int* dims = [self.n_var, self.n_var]

            # Misc.
            double OW[1]     # Objective type: we currently only support single objective

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()
        if not PyArray_ISCARRAY(y): y = y.copy()

        # Determine room for Hessian and multiplier sign.
        nnzh = self.get_nnzh()
        OW[0] = obj_weight if self.objtype == 0 else -obj_weight

        # Allocate storage and evaluate Hessian.
        H = np.empty(nnzh, dtype=np.double)
        ampl_sphes(self.asl, <double*>H.data, -1, OW, <double*>y.data)

        if coord: # return Hesian in coordinate format.
            a_icol = np.empty(nnzh, dtype=np.int)
            a_irow = np.empty(nnzh, dtype=np.int)

            k = 0
            for i in range(self.n_var):
                j0 = self.asl.i.sputinfo_.hcolstarts[i  ]
                j1 = self.asl.i.sputinfo_.hcolstarts[i+1]
                for j in range(j0,j1):
                    a_irow.data[k] = self.asl.i.sputinfo_.hrownos[j]
                    a_icol.data[k] = i
                    k += 1

            # Return the triple (J, irow, icol).
            return (H, a_irow, a_icol)

        else: # return Hessian in LL format.

            if spHess is None:
                spHess = SpMatrix_NewLLMatObject(dims, SYMMETRIC,
                                                 nnzh, store_zeros)
                if spHess is None:
                    raise ValueError

            for i in range(self.n_var):
                j0 = self.asl.i.sputinfo_.hcolstarts[i  ]
                j1 = self.asl.i.sputinfo_.hcolstarts[i+1]
                for j in range(j0,j1):
                    SpMatrix_LLMatSetItem(<void*>spHess, i,
                                          self.asl.i.sputinfo_.hrownos[j],
                                          H[j])
            return spHess
                
    def H_prod(self, ndarray[np.double_t] y, ndarray[np.double_t] v,
               double obj_weight=1.0):
        """Compute matrix-vector product Hv of Lagrangian Hessian
        times a vector."""

        cdef:
            double OW[1]
            ndarray[np.double_t] Hv
            
        # Ensure contiguous input.
        if not PyArray_ISCARRAY(y): y = y.copy()
        if not PyArray_ISCARRAY(v): v = v.copy()

        OW[0] = obj_weight if self.objtype == 0 else -obj_weight
        Hv = np.empty(self.n_var, dtype=np.double)
        
        # Evaluate matrix-vector product Hv
        ampl_hvcomp(self.asl, <double*>Hv.data, <double*>v.data,
                    -1, OW, <double*>y.data)

        return Hv
    

    def gHi_prod(self, ndarray[np.double_t] g, ndarray[np.double_t] v):
        """Compute the vector of dot products (g,Hi*v) of with the
        constraint Hessians."""

        cdef:
            ndarray[np.double_t] gHiv = np.empty(self.n_con, type=np.double)
            ndarray[np.double_t] hv = np.empty(self.n_var, type=np.double)
            ndarray[np.double_t] y = np.zeros(self.n_con, type=np.double)

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(g): g = g.copy()
        if not PyArray_ISCARRAY(v): v = v.copy()

        # Skip linear constraints.
        gHiv[self.nlc:self.n_con] = 0

        # Process nonlinear constraints.
        for i in range(self.nlc):
            # Set vector of multipliers to (0, 0, ..., -1, ..., 0).
            y[i] = -1.0

            # Compute Hi * v by setting OW to NULL.
            ampl_hvcomp(self.asl, <double*>hv.data, <double*>v.data,
                        -1, NULL, <double*>y.data);

            # Compute dot product (g, Hi*v). Should use BLAS.
            gHiv[i] = np.dot(hv, g)

            # Reset i-th multiplier.
            y[i] = 0

        return gHiv

    def set_x(self, ndarray[np.double_t] x):
        """Declare x as current primal value."""

        # Call xknown() with given x as argument, to prevent subsequent
        # calls to objval, objgrad, etc., to check whether their argument
        # has changed since the last call. Users must not forget to call
        # Unset_x when they are finished, and before changing the value
        # of x, or to call Set_x again with an updated value of x.

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(g): g = g.copy()

        ampl_xknown(self.asl, <double*>x.data)

    def unset_x(self):
        """Release current primal value."""
        self.asl.i.x_known = 0
    
    def ampl_sol(self, ndarray[np.double_t] x, ndarray[np.double_t] y, msg):
        """Write primal and dual solution."""

        # Ensure contiguous input.
        if not PyArray_ISCARRAY(x): x = x.copy()
        if not PyArray_ISCARRAY(y): y = y.copy()

        # Suppress message echo, force .sol writing.
        self.Oinfo.wantsol = 9

        # Output solution.
        write_sol_ASL(self.asl, msg, <double*>x.data, <double*>y.data, &self.Oinfo)

        # Flag that a solution has been written.
        self.ampl_written_sol = True
