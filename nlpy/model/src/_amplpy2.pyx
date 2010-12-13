#import numpy as np
import os.path
cimport cpython
cimport libc.stdio
cimport numpy as np

#@cython.boundscheck(False)
cdef enum:
    ASL_read_f    = 1
    ASL_read_fg   = 2
    ASL_read_fgh  = 3
    ASL_read_pfg  = 4
    ASL_read_pfgh = 5
    want_xpi0     = 3

cdef extern from "amplutils.h":
    ctypedef struct ASL:
        pass
    ASL* ASL_alloc(int)
    void* ASL_free(ASL**)
    libc.stdio.FILE* jac0dim_ASL(ASL*, char*, int)
    int pfgh_read_ASL(ASL*, libc.stdio.FILE*, int)

    int ampl_get_n_var(ASL*)
    int ampl_get_nbv(ASL*)
    int ampl_get_niv(ASL*)
    int ampl_get_n_con(ASL*)
    int ampl_get_n_obj(ASL*)
    int ampl_get_nlo(ASL*)
    int ampl_get_nranges(ASL*)
    int ampl_get_nlc(ASL*)
    int ampl_get_nlnc(ASL*)
    int ampl_get_nlvb(ASL*)
    int ampl_get_nlvbi(ASL*)
    int ampl_get_nlvc(ASL*)
    int ampl_get_nlvci(ASL*)
    int ampl_get_nlvo(ASL*)
    int ampl_get_nlvoi(ASL*)
    int ampl_get_lnc(ASL*)
    int ampl_get_nzc(ASL*)
    int ampl_get_nzo(ASL*)
    int ampl_get_maxrownamelen(ASL*)
    int ampl_get_maxcolnamelen(ASL*)

    void ampl_set_n_conjac(ASL* asl, int val0, int val1)
    void ampl_get_n_conjac(ASL* asl, int *val0, int *val1)
    void ampl_set_want_xpi0(ASL* asl, int val)
    int ampl_get_objtype(ASL* asl, int nobj)
    int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b)

cdef class ampl:

    cdef ASL* asl
    cdef libc.stdio.FILE* ampl_file

    def __cinit__(self):
        """cinit is called before init; allocates the ASL structure."""
        self.asl = ASL_alloc(ASL_read_pfgh)
        if self.asl is NULL:
            cpython.PyErr_NoMemory()

    def __dealloc__(self):
        """Free the ASL structure."""
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
        
        self.ampl_file = jac0dim_ASL(asl, stub, len(stub)) # open stub
        ampl_set_want_xpi0(asl, want_xpi0)     # ask for initial x and pi
        pfgh_read_ASL(asl, self.ampl_file, 0)  # read the stub

    # These simple routines allow us to access complicated ASL
    # structure elements via AMPL's macros.
    def get_n_var(self): return ampl_get_n_var(self.asl)
    def get_nbv(self): return ampl_get_nbv(self.asl)
    def get_niv(self): return ampl_get_niv(self.asl)
    def get_n_con(self): return ampl_get_n_con(self.asl)
    def get_n_obj(self): return ampl_get_n_obj(self.asl)
    def get_nlo(self): return ampl_get_nlo(self.asl)
    def get_nranges(self): return ampl_get_nranges(self.asl)
    def get_nlc(self): return ampl_get_nlc(self.asl)
    def get_nlnc(self): return ampl_get_nlnc(self.asl)
    def get_nlvb(self): return ampl_get_nlvb(self.asl)
    def get_nlvbi(self): return ampl_get_nlvbi(self.asl)
    def get_nlvc(self): return ampl_get_nlvc(self.asl)
    def get_nlvci(self): return ampl_get_nlvci(self.asl)
    def get_nlvo(self): return ampl_get_nlvo(self.asl)
    def get_nlvoi(self): return ampl_get_nlvoi(self.asl)
    def get_lnc(self): return ampl_get_lnc(self.asl)
    def get_nzc(self): return ampl_get_nzc(self.asl)
    def get_nzo(self): return ampl_get_nzo(self.asl)
    def get_maxrownamelen(self): return ampl_get_maxrownamelen(self.asl)
    def get_maxcolnamelen(self): return ampl_get_maxcolnamelen(self.asl)

    def get_objtype(self, nobj):
        """Determine if a problem is maximization or minimization."""
        return ampl_get_objtype(self.asl, nobj)

    def get_dim(self):
        """Obtain n and m."""
        return (self.get_n_var(), self.get_n_con())
    
    ## def get_nnzh(self):
    ##     """Get the number of nonzeros in the Lagrangian Hessian."""
    ##     cdef int nnzh, nnzj
    ##     ampl_get_dims(self.asl, &nnzj, &nnzh)
    ##     return (nnzh, nnzj)
    
