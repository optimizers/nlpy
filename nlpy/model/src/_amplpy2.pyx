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

    def set_want_xpi0(self, val): ampl_set_want_xpi0(self.asl, val)

    cdef get_n_var(self):
    cdef get_nbv(self):
    cdef get_niv(self):
    cdef get_n_con(self):
    cdef get_n_obj(self):
    cdef get_nlo(self):
    cdef get_nranges(self):
    cdef get_nlc(self):
    cdef get_nlnc(self):
    cdef get_nlvb(self):
    cdef get_nlvbi(self):
    cdef get_nlvc(self):
    cdef get_nlvci(self):
    cdef get_nlvo(self):
    cdef get_nlvoi(self):
    cdef get_lnc(self):
    cdef get_nzc(self):
    cdef get_nzo(self):
    cdef get_maxrownamelen(self):
    cdef get_maxcolnamelen(self):

    def get_objtype(self, nobj): return ampl_get_objtype(self.asl, nobj)
    
    ## def get_nnzh(self):
    ##     """Get the number of nonzeros in the Lagrangian Hessian."""
    ##     cdef int nnzh, nnzj
    ##     ampl_get_dims(self.asl, &nnzj, &nnzh)
    ##     return (nnzh, nnzj)
    
