#import numpy as np
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

    int  ampl_get_n_var(ASL*)
    int  ampl_get_n_con(ASL*)
    void ampl_set_want_xpi0(ASL*, int)
    int  ampl_get_objtype(ASL*, int)
    int  ampl_get_nnzj(ASL*)
    void  ampl_get_dims(ASL*, int*, int*)

cdef class ampl:

    cdef ASL* asl
    cdef libc.stdio.FILE* ampl_file
    cdef public int m, n
    
    def __cinit__(self):
        self.asl = ASL_alloc(ASL_read_pfgh)
        if self.asl is NULL:
            cpython.PyErr_NoMemory()

    def __dealloc__(self):
        if self.asl is not NULL:
            ASL_free(&self.asl)

    def __init__(self, stub):
        """Initialize an ampl object."""
        asl = self.asl

        # Check that the file is readable before giving it to ampl.
        try:
            open(stub,'r')
        except IOError:
            print 'Could not open file %s' % stub
        
        self.ampl_file = jac0dim_ASL(asl, stub, len(stub)) # open stub
        ampl_set_want_xpi0(asl, want_xpi0)     # get initial x and pi
        pfgh_read_ASL(asl, self.ampl_file, 0)  # read the stub

        # Save the problem dimensions.
        ampl_get_dims(asl, &self.n, &self.m)

    def get_m_con(self):
        """Get the number of constraints."""
        return ampl_get_n_con(self.asl)

    def set_want_xpi0(self, val):
        ampl_set_want_xpi0(self.asl, val)

    def get_objtype(self, nobj):
        """Determine whether a problem is minimization or
        maximization."""
        return ampl_get_objtype(self.asl, nobj)
    
    def get_nnzj(self):
        """Get the number of nonzeros in the Jacobian."""
        return ampl_get_nnzj(self.asl)

    ## def get_nnzh(self):
    ##     """Get the number of nonzeros in the Lagrangian Hessian."""
    ##     cdef int nnzh, nnzj
    ##     ampl_get_dims(self.asl, &nnzj, &nnzh)
    ##     return (nnzh, nnzj)
    
