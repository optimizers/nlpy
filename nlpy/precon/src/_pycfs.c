
/* ========================================================================== */

/*
 *                                   P y c f s
 *
 *   P y t h o n   i n t e r f a c e   t o   t h e   I C F S   p a c k a g e
 */

/* ========================================================================== */

/* Indicate that module is split across several files */
#define PY_ARRAY_UNIQUE_SYMBOL icfs

#include "Python.h"                /* Main Python header file */
#include "arrayflavor.h"
#include "spmatrix.h"
#include "spmatrix_api.h"
#include "ll_mat.h"
#include <time.h>

/* For spawning new LLMatObjects */
#define  SYMMETRIC 1
#define  GENERAL   0

/* ========================================================================== */

/*
 *    D e f i n i t i o n   o f   P y c f s   c o n t e x t   o b j e c t
 */

/* ========================================================================== */

staticforward PyTypeObject PycfsType;     /* Precise definition appears below */

typedef struct PycfsObject {
    PyObject_VAR_HEAD
    int n;                                                     /* Matrix size */
    int nnz;                                            /* Number of nonzeros */
    int p;                    /* Amount of additional memory to derive factor */
    double shift;      /* Shift necessary to complete Cholesky factorizations */
    double *ldiag;                     /* Incomplete Cholesky diagonal values */
    double *l;              /* Incomplete Cholesky factor off-diagonal values */
    int *lcolptr, *lrowind;       /* Incomplete Cholesky factor in CSC format */
    double  *diag;                             /* Diagonal of original matrix */
    double  *val;                          /* Off-diagonal of original matrix */
    int *rowind, *colind, *colptr;           /* Original matrix in CSC format */
    double *neg_curv;
    double  nc;
} PycfsObject;

/* ========================================================================== */

/* Support different Fortran compilers */
#ifdef _AIX
#define FUNDERSCORE(a) a
#else
#define FUNDERSCORE(a) a##_
#endif

#define SRTDAT2  FUNDERSCORE(srtdat2)
#define DICFS    FUNDERSCORE(dicfs)
#define DPCG     FUNDERSCORE(dpcg)
#define DSSYAX   FUNDERSCORE(dssyax)
#define DDOT     FUNDERSCORE(ddot)
#define DNRM2    FUNDERSCORE(dnrm2)

#define ZERO     (double)0.0
#define PYCFS_MAX(a,b) ((a) > (b)) ? (a) : (b)

/* ========================================================================== */

DL_EXPORT( void ) init_pycfs( void );
static PyObject *Pycfs_icfs( PyObject *self, PyObject *args );
static PyObject *Pycfs_get_shift( PycfsObject *self, PyObject *args );
static PyObject *Pycfs_pcg(  PycfsObject *self, PyObject *args );
static PyObject *Pycfs_fetch( PycfsObject *self, PyObject *args );

static void Pycfs_dealloc( PycfsObject *self );
static PyObject *Pycfs_getattr( PycfsObject *self, char *name );
static PyObject *NewPycfsObject( LLMatObject *llmat, int p );

extern void SRTDAT2( int *n, int *nnz, double *a, double *adiag,
		     int *row_ind, int *col_ind, int *col_ptr, int *iwa);

extern void DICFS( int *n, int *nnz, double *a, double *adiag,
		   int *acol_ptr, int *arow_ind, double *l, double *ldiag,
		   int *lcol_ptr, int *lrow_ind, int *p, double *alpha,
		   int *iwa, double *wa1, double *wa2 );

extern void DPCG( int *n, double *a, double *adiag, int *acol_ptr,
		  int *arow_ind, double *l, double *ldiag, int *lcol_ptr,
		  int *lrow_ind, double *b, double *rtol, int *itmax,
		  double *x, int *iters, int *info, double *p,
		  double *q, double *r, double *z );

extern void DSSYAX( int *n, double *a, double *adiag, int *jptr,
		    int *indr, double *x, double *y );

extern double DDOT( int *n, double *dx, int *incx, double *dy, int *incy );

extern double DNRM2( int *n, double *x, int *incx );

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */

static PyObject *NewPycfsObject( LLMatObject *llmat, int p ) {

    PycfsObject *self;
    int          n, nnz;
    int          i, k, elem;
    int         *iwa;                 /* Int work array for Fortran routine */
    double      *wa1, *wa2;           /* Double work arrays */

    /* Create new instance of object */
    if( !(self = PyObject_New( PycfsObject, &PycfsType ) ) )
	    return PyErr_NoMemory( );

    n   = llmat->dim[0];
    nnz = llmat->nnz;

    /* Fill in object -- Part I */
    self->n   = n;
    self->nnz = nnz;
    self->p   = p;
    self->nc  = 0;

    if( !( self->val = (double *)malloc( nnz * sizeof( double ) ) ) )
	    return NULL;
    if( !( self->rowind = (int *)malloc( nnz * sizeof( double ) ) ) )
	    return NULL;
    if( !( self->colind = (int *)malloc( nnz * sizeof( double ) ) ) )
	    return NULL;

    /* Get matrix in coordinate format. Adjust indices to be 1-based. */
    elem = 0;
    for( i = 0; i < n; i++ ) {
	    k = llmat->root[i];
	    while( k != -1 ) {
	        self->rowind[ elem ] = i + 1;
	        self->colind[ elem ] = llmat->col[k] + 1;
	        self->val[ elem ] = llmat->val[k];
	        k = llmat->link[k];
	        elem++;
	    }
    }

    if( !( self->diag = (double *)malloc( n * sizeof( double ) ) ) )
	    return NULL;
    if( !( self->colptr = (int *)malloc( (n+1) * sizeof( int ) ) ) )
	    return NULL;

    /* Allocate Fortran work array */
    if( !( iwa = (int *)malloc( n * sizeof( int ) ) ) )
	    return NULL;

    /* Reorder matrix in sparse compressed column format and
     * obtain diagonal in separate array as a dense vector. */
    SRTDAT2( &(self->n), &(self->nnz), self->val, self->diag, self->rowind,
	     self->colind, self->colptr, iwa );

    free( iwa );

    /* Fill in object -- Part II */
    if( !(self->l = (double *)malloc( (nnz + n*p) * sizeof( double ) ) ) )
      return NULL;
    if( !(self->ldiag = (double *)malloc( n * sizeof( double ) ) ) )
      return NULL;
    if( !(self->lcolptr = (int *)malloc( (n+1) * sizeof( double ) ) ) )
      return NULL;
    if( !(self->lrowind = (int *)malloc( (nnz + n*p) * sizeof( double ) ) ) )
      return NULL;

    /* Allocate work arrays for dicfs */
    if( !( iwa = (int *)malloc( 3*n * sizeof( int ) ) ) )
      return NULL;
    if( !( wa1 = (double *)malloc( n * sizeof( double ) ) ) )
      return NULL;
    if( !( wa2 = (double *)malloc( n * sizeof( double ) ) ) )
      return NULL;

    /* If limited memory < 0, return square root of diagonal */
    self->shift = ZERO;
    if( p < 0 ) {
      for( i = 0; i < n; i++ ) {
        self->ldiag[i]   = sqrt( self->diag[i] );
        self->lcolptr[i] = 1;
      }
      self->lcolptr[n] = 1;
    } else {
      /* Obtain incomplete Cholesky factor L */
      DICFS( &(self->n), &(self->nnz), self->val, self->diag, self->colptr,
             self->rowind, self->l, self->ldiag, self->lcolptr, self->lrowind,
             &(self->p), &(self->shift), iwa, wa1, wa2 );
    }

    free( self->colind );
    free( iwa ); free( wa1 ); free( wa2 );

    return (PyObject *)self;
}

/* ========================================================================== */

static char Pycfs_pcg_Doc[] = "Solve Ax=b using a preconditioned conjugate gradient";

static PyObject *Pycfs_pcg( PycfsObject *self, PyObject *args ) {

    //int            dim[1];

    PyArrayObject *a_x, *a_b, *a_d;
    //PyObject      *b, *xx, *dd;
    double        *pb;                  /* Right-hand side */
    double        *x, *d;
    double         rtol = 1.0e-6, relres = ZERO, normb;
    int            n, iters, info, maxiter, incx = 1;
    double        *wa2, *wa3, *wa4;   /* Double work arrays */
    int            i;
    clock_t        t_solve = (clock_t)0;
    double         etime;

    /* We read a right-hand side b */
    if( !PyArg_ParseTuple( args, "O!O!O!|id", &PyArray_Type, &a_b,
                                              &PyArray_Type, &a_x,
                                              &PyArray_Type, &a_d,
                                              &maxiter, &rtol ) )
	return NULL;
    if( a_b->descr->type_num  != tFloat64 ) return NULL;
    if( a_x->descr->type_num  != tFloat64 ) return NULL;
    if( a_d->descr->type_num  != tFloat64 ) return NULL;

    if( !a_b ) return NULL;                         /* conversion error */
    if( a_b->nd != 1 ) return NULL;          /* b must have 1 dimension */
    if( a_b->dimensions[0] != self->n ) return NULL;      /* and size n */
    if( !a_x ) return NULL;                         /* conversion error */
    if( a_x->nd != 1 ) return NULL;          /* x must have 1 dimension */
    if( a_x->dimensions[0] != self->n ) return NULL;      /* and size n */
    if( !a_d ) return NULL;                         /* conversion error */
    if( a_d->nd != 1 ) return NULL;          /* d must have 1 dimension */
    if( a_d->dimensions[0] != self->n ) return NULL;      /* and size n */
    //PyArray_XDECREF( a_b );
    pb = (double *)a_b->data;
    x  = (double *)a_x->data;
    d  = (double *)a_d->data;
    
    /* Set max number of iterations */
    if( maxiter <= 0 ) maxiter = self->n;
    if( maxiter < 100 ) maxiter = 100;

    n = self->n;

    self->neg_curv = (double *)a_d->data;

    /* Initalize work arrays */
    if( !( wa2 = (double *)malloc( n * sizeof( double ) ) ) )
	    return NULL;
    if( !( wa3 = (double *)malloc( n * sizeof( double ) ) ) )
	    return NULL;
    if( !( wa4 = (double *)malloc( n * sizeof( double ) ) ) )
	    return NULL;

    iters = 0; info = 0;

    /* Solve */
    t_solve = clock();
    DPCG( &(self->n), self->val, self->diag, self->colptr, self->rowind, self->l,
	  self->ldiag, self->lcolptr, self->lrowind, pb, &rtol, &maxiter, x,
	  &iters, &info, self->neg_curv, wa2, wa3, wa4 );

    if( info == 2 ) { /* Negative curvature was detected */
	    DSSYAX( &(self->n), self->val, self->diag, self->colptr, self->rowind,
		        self->neg_curv, wa4 );
	    self->nc = DDOT( &(self->n), self->neg_curv, &(incx), wa4, &(incx) ) /
	               DNRM2( &(self->n), self->neg_curv, &(incx) );
    }

    t_solve = clock() - t_solve;
    etime = (double)t_solve/CLOCKS_PER_SEC;

    free( wa2 ); free( wa3 );

    /* Compute relative residual: wa4 = ||Ax-b||/||b|| */
    DSSYAX( &(self->n), self->val, self->diag, self->colptr, self->rowind, x, wa4 );
    for( i=0; i<n; i++ ) wa4[i] -= pb[i];
    normb  = DNRM2( &(self->n), pb, &(incx) );
    relres = DNRM2( &(self->n), wa4, &(incx) );
    if( normb > ZERO ) relres = relres/normb;

    free( wa4 );

    return Py_BuildValue("iiddd", iters, info, relres, self->nc, etime);
}

/* ========================================================================== */

static char Pycfs_get_shift_Doc[] = "Retrieve final shift value";

static PyObject *Pycfs_get_shift( PycfsObject *self, PyObject *args ) {

    return PyFloat_FromDouble( self->shift );
}

/* ========================================================================== */

static char Pycfs_fetch_Doc[] = "Fetch incomplete Cholesky factor in LL format";

static PyObject *Pycfs_fetch( PycfsObject *self, PyObject *args ) {

    LLMatObject *L;
    int          dim[2];
    int          nnzL;
    int          i, j;

    dim[0] = self->n;
    dim[1] = self->n;
    nnzL = self->lcolptr[ self->n ] - 1;
    //printf( " dim = [%-d, %-d], nnzL = %-d\n", dim[0], dim[1], nnzL );
    /* L is a lower triangular matrix */
    L = (LLMatObject *)SpMatrix_NewLLMatObject( dim, GENERAL, nnzL + self->n );
    
    // debug
    //printf( " lcolptr = [ " );
    //for( i = 0; i < self->n + 1; i++ )
    //    printf( "%-d ", self->lcolptr[i] );
    //printf( "]\n" );
    //printf( " lrowind = [ " );
    //for( i = 0; i < nnzL; i++ )
    //    printf( "%-d ", self->lrowind[i] );
    //printf( "]\n" );
    //printf( " l = [ " );
    //for( i = 0; i < nnzL; i++ )
    //    printf( "%-g ", self->l[i] );
    //printf( "]\n" );
    //printf( " ldiag = [ " );
    //for( i = 0; i < self->n; i++ )
    //    printf( "%-g ", self->ldiag[i] );
    //printf( "]\n" );

    /* Fill strict lower triangle of L */
    for( j = 0; j < self->n; j++ )
        for( i = self->lcolptr[j]; i < self->lcolptr[j+1]; i++ ) {
            //printf( " i = %-d, rowind[%-d] = %-d, j = %-d, val = %-f\n",
            //        i-1, i-1, self->lrowind[i-1]-1, j, self->l[i-1] );
            /* Account for Fortran 1-based indexing */
            SpMatrix_LLMatSetItem( L, self->lrowind[i-1]-1, j, self->l[i-1] );
        }

    /* Fill diagonal of L */
    for( j = 0; j < self->n; j++ ) {
        //printf( " i = %-d, j = %-d, val = %-g\n", j, j, self->ldiag[j] );
        SpMatrix_LLMatSetItem( L, j, j, self->ldiag[j] );
    }

    return (PyObject *)L;
}

/* ========================================================================== */

/* This is necessary as Pycfs_pcg takes a PycfsObject* as argument */

static PyMethodDef Pycfs_special_methods[] = {
    { "pcg",   (PyCFunction)Pycfs_pcg,   METH_VARARGS, Pycfs_pcg_Doc   },
    { "fetch", (PyCFunction)Pycfs_fetch, METH_VARARGS, Pycfs_fetch_Doc },
    { "get_shift", (PyCFunction)Pycfs_get_shift, METH_VARARGS, Pycfs_get_shift_Doc },
    { NULL,    NULL,                     0,            NULL            }
};

/* ========================================================================== */

static char Pycfs_icfs_Doc[] = "Compute incomplete Cholesky preconditioner";

static PyObject *Pycfs_icfs( PyObject *self, PyObject *args ) {

    /* Input must be the lower triangle of a symmetric matrix */

    PyObject      *mat;                   /* Input matrix */
    int            p;                     /* limited memory */

    /* Read input matrix and limited memory factor */
    if( !PyArg_ParseTuple( args, "Oi", &mat, &p ) )
	    return NULL;

    /* Spawn new Pycfs Object */
    return NewPycfsObject( (LLMatObject *)mat, p );
}

/* ========================================================================== */

/*
 *       D e f i n i t i o n   o f   P y c f s   c o n t e x t   t y p e
 */

/* ========================================================================== */

static PyTypeObject PycfsType = {
    PyObject_HEAD_INIT(NULL)
    0,
    "pycfs_context",
    sizeof(PycfsObject),
    0,
    (destructor)Pycfs_dealloc,  /* tp_dealloc */
    0,                          /* tp_print */
    (getattrfunc)Pycfs_getattr, /* tp_getattr */
    0,				/* tp_setattr */
    0,				/* tp_compare */
    0,				/* tp_repr */
    0,				/* tp_as_number*/
    0,				/* tp_as_sequence*/
    0,				/* tp_as_mapping*/
    0,				/* tp_hash */
};

/* ========================================================================== */

/*
 *            D e f i n i t i o n   o f   P y c f s   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef PycfsMethods[] = {
    { "icfs",   Pycfs_icfs,   METH_VARARGS, Pycfs_icfs_Doc  },
    { NULL,     NULL,         0,            NULL            }
};

/* ========================================================================== */

static void Pycfs_dealloc( PycfsObject *self ) {

    free( self->val );     free( self->diag );
    free( self->rowind );  /* free( self->colind ); */
    free( self->l );       free( self->ldiag );
    free( self->lcolptr ); free( self->lrowind );
    free( self->colptr );  //free( self->neg_curv);

    PyObject_Del(self);
}

/* ========================================================================== */

static PyObject *Pycfs_getattr( PycfsObject *self, char *name ) {

    if( strcmp( name, "shape" ) == 0 )
	return Py_BuildValue( "(i,i)", self->n, self->n );
    if( strcmp( name, "nnz" ) == 0 )
	return Py_BuildValue( "i", self->nnz );
    if( strcmp( name, "__members__" ) == 0 ) {
	char *members[] = {"shape", "nnz"};
	int i;

	PyObject *list = PyList_New( sizeof(members)/sizeof(char *) );
	if( list != NULL ) {
	    for( i = 0; i < sizeof(members)/sizeof(char *); i++ )
		PyList_SetItem( list, i, PyString_FromString(members[i]) );
	    if( PyErr_Occurred() ) {
		Py_DECREF( list );
		list = NULL;
	    }
	}
	return list;
    }
    return Py_FindMethod( Pycfs_special_methods, (PyObject *)self, name );
}

/* ========================================================================== */

DL_EXPORT( void ) init_pycfs( void ) {

    PyObject *m, *d;

    PycfsType.ob_type = &PyType_Type;

    m = Py_InitModule3( "_pycfs", PycfsMethods, "Python interface to ICFS" );
    /* m = Py_InitModule( "_pycfs", PycfsMethods ); */

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "PycfsType", (PyObject *)&PycfsType);

    import_array( );         /* Initialize the Numarray module. */
    import_spmatrix( );      /* Initialize the PySparse module. */

    /* Check for errors */
    if (PyErr_Occurred())
	Py_FatalError("Unable to initialize module pycfs");

    return;
}

/* ========================================================================== */

/*
 *                      E n d   o f   m o d u l e   P y c f s
 */

/* ========================================================================== */
