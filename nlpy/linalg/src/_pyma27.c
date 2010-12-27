
/* ========================================================================== */

/*
 *                                   P y m a 2 7
 *
 *   P y t h o n   i n t e r f a c e   t o   t h e   M A 2 7   p a c k a g e
 */

/* ========================================================================== */

/*
   $Revision: 84 $
   $Date: 2008-09-14 22:49:23 -0400 (Sun, 14 Sep 2008) $
*/

/* Indicate that module is split across several files */
#define PY_ARRAY_UNIQUE_SYMBOL ma27

#include <Python.h>                /* Main Python header file */
#include <cblas.h>
#include "numpy/arrayobject.h"     /* NumPy header */
#include "spmatrix.h"
#include "spmatrix_api.h"
#include "ll_mat.h"
#include "ma27.h"

#define  SYMMETRIC 1  /* for SpMatrix_NewLLMatObject() */

/* ========================================================================== */

/*
 *    D e f i n i t i o n   o f   P y m a 2 7   c o n t e x t   o b j e c t
 */

/* ========================================================================== */

staticforward PyTypeObject Pyma27Type;    /* Precise definition appears below */

typedef struct Pyma27Object {
    PyObject_VAR_HEAD
    Ma27_Data  *data;
    double     *a;
} Pyma27Object;

#define Pyma27Object_Check(v)  ((v)->ob_type == &Pyma27Type)

/* ========================================================================== */

DL_EXPORT( void ) init_pyma27( void );
static PyObject     *Pyma27_ma27(       Pyma27Object *self,  PyObject *args );
static PyObject     *Pyma27_refine(     Pyma27Object *self,  PyObject *args );
static PyObject     *Pyma27_factor(     PyObject     *self,  PyObject *args );
static void          Pyma27_dealloc(    Pyma27Object *self                  );
static PyObject     *Pyma27_getattr(    Pyma27Object *self,  char *name     );
static PyObject     *Pyma27_Stats(      Pyma27Object *self,  PyObject *args );
static PyObject     *Pyma27_fetch_perm( Pyma27Object *self                  );
static PyObject     *Pyma27_fetch_lb(   Pyma27Object *self,  PyObject *args );
static Pyma27Object *NewPyma27Object(   LLMatObject  *llmat, PyObject *sqd  );
extern PyObject *newCSRMatObject(int dim[], int nnz);
void coord2csr( int n, int nz, int *irow, int *jcol, double *val,
                int *iptr, int *jind, double *xval );

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */


static Pyma27Object *NewPyma27Object( LLMatObject *llmat, PyObject *sqd ) {

    Pyma27Object *self;
    int          n  = llmat->dim[0],
                 nz = llmat->nnz;;
    int          i, k, elem;
    int          error;

    /* Create new instance of object */
    if( !(self = PyObject_New( Pyma27Object, &Pyma27Type ) ) )
        return NULL; //PyErr_NoMemory( );

    self->data = Ma27_Initialize( nz, n, NULL );

    /* Set pivot-for-stability threshold is matrix is SQD */
    if( sqd == Py_True ) self->data->cntl[0] = 1.0e-15;

    /* Keep a copy of matrix in a in case memory needs to be adjusted */
    /* Array a is never altered; we work with factors */
    self->a = (double *)NLPy_Calloc( nz, sizeof(double) );

    /* Get matrix in coordinate format. Adjust indices to be 1-based. */
    elem = 0;
    for( i = 0; i < n; i++ ) {
        k = llmat->root[i];
        while( k != -1 ) {
            self->data->irn[ elem ] = i + 1;
            self->data->icn[ elem ] = llmat->col[k] + 1;
            self->a[ elem ] = llmat->val[k];
            k = llmat->link[k];
            elem++;
        }
    }

    /* Analyze */
    error = Ma27_Analyze( self->data, 0 ); // iflag = 0: automatic pivot choice
    if( error ) {
        fprintf( stderr, " Error return code from Analyze: %-d\n", error );
        return NULL; //Py_None; // ----- ADJUST ----- ?
    }

    /* Factorize */
    error = Ma27_Factorize( self->data, self->a );
    if( error ) {
        fprintf( stderr, " Error return code from Factorize: %-d\n", error );
        return NULL; //Py_None; // ----- ADJUST ----- ?
    }

    /* Find out if matrix was rank deficient */
    self->data->rankdef = 0;
    self->data->rank = self->data->n;
    if( self->data->info[0] == 3 || self->data->info[0] == -5 ) {
        self->data->rankdef = 1;
        self->data->rank = self->data->info[1];
    }

    return self;
}

/* ========================================================================== */

static char Pyma27_fetch_perm_Doc[] = "Fetch variables permutation computed by MA27";

static PyObject *Pyma27_fetch_perm( Pyma27Object *self ) {

    PyObject *perm;
    int       i, err;

    /* New list to hold the pivot order */
    perm = PyList_New( 0 );
    if( !perm ) return NULL;
    for( i = 0; i < self->data->n; i++ ) {
        err = PyList_Append( perm,
                             (PyObject *)PyInt_FromLong(self->data->ikeep[i]));
        if( err < 0 ) {
            Py_DECREF( perm );
            return NULL;
        }
    }

    return Py_BuildValue( "O", perm );
}

/* ========================================================================== */

static char Pyma27_fetch_lb_Doc[] = "Fetch factors of A computed by MA27";

static PyObject *Pyma27_fetch_lb( Pyma27Object *self, PyObject *args ) {

    PyObject     *Lmat, *Dmat;
    LLMatObject  *L, *D;
    double       *l, *d;
    int          *id, *jd, *il, *jl;  /* L and D^{-1} in coordinate format */
    int           nnzL, nnzD, nblk, n1x1, n2x2, latop, liwm1, *iwp1, *colrhs;
    int           i;
    int           n = self->data->n;
    //int *inv_perm, ii, jj;

    if( !PyArg_ParseTuple( args, "OO", &Lmat, &Dmat ) ) return NULL;
    L = (LLMatObject *)Lmat;
    D = (LLMatObject *)Dmat;

    /* Obtain the number of nonzeros in factors L and D^{-1} */
    nblk = abs( self->data->iw[0] );   /* # block pivots */
    if( nblk == 0 ) {
        Py_INCREF( Py_None );
        return Py_None;
    }
    iwp1 = (self->data->iw)+1;
    liwm1 = self->data->liw - 1;
    MA27QDEMASC( &n, iwp1, &liwm1, self->data->iw1,
                 &nblk, &latop, self->data->icntl );

    n2x2 = self->data->info[13];   /* No. of 2x2 pivots. */
    n1x1 = n - 2*n2x2;             /* No. of 1x1 pivots. */
    nnzD = n1x1 + 4*n2x2;          /* 1 nz for each 1x1, 4 nz's for each 2x2. */
    nnzL = n + latop;              /* An upper bound only, not exact. */

    //printf( " It would appear that nnzl = %-d, nnzd = %-d, latop = %-d, n2x2 = %-d, nblk = %-d\n",
     //       nnzl, nnzd, latop, n2x2, nblk );

    /* Allocate space for D^{-1} and L */
    d  = (double *)NLPy_Calloc( nnzD, sizeof(double) );
    l  = (double *)NLPy_Calloc( nnzL, sizeof(double) );
    id = (int *)NLPy_Calloc( nnzD, sizeof(int) );
    jd = (int *)NLPy_Calloc( nnzD, sizeof(int) );
    il = (int *)NLPy_Calloc( nnzL, sizeof(int) );
    jl = (int *)NLPy_Calloc( nnzL, sizeof(int) );

    colrhs = (int *)NLPy_Calloc( self->data->maxfrt, sizeof(int) );

    /* Obtain lower triangle of D^{-1} and upper triangle of L */
    MA27FACTORS( &n, self->data->factors, &(self->data->la),
                 iwp1, &liwm1, &(self->data->maxfrt), self->data->iw1,
                 &nblk, &latop, self->data->icntl, colrhs,
                 &nnzD, id, jd, d,
                 &nnzL, il, jl, l );

    /*
    if( !(inv_perm = (int *)malloc( self->data->n * sizeof( int ) )) ) {
        printf( " Oops! Can't malloc inv_perm\n" );
        return NULL;
    }
    for( i = 0; i < self->data->n; i++ )
        inv_perm[ self->data->ikeep[i]-1 ] = i+1;

    printf( "inv_perm = [" );
    for( i = 0; i < self->data->n; i++ )
        printf( " %-d ", inv_perm[i] );
    printf( "]\n" );
    */

    /* At this point, nnzL is exact.  Build sparse matrices D^{-1}
     * and L.  Account for 0-based indexing */
    for( i = 0; i < nnzL; i++ ) {
        //ii = inv_perm[ il[i]-1 ];
        //jj = inv_perm[ jl[i]-1 ];
        //ii = self->data->ikeep[ il[i]-1 ];
        //jj = self->data->ikeep[ jl[i]-1 ];
        //SpMatrix_LLMatSetItem( L, jj-1, ii-1, l[i] );
        SpMatrix_LLMatSetItem( L, jl[i]-1, il[i]-1, l[i] );
    }

    for( i = 0; i < nnzD; i++ )
        SpMatrix_LLMatSetItem( D, id[i]-1, jd[i]-1, d[i] );

    /* Build L in compressed sparse row format */
    /*
    Lcsr = (CSRMatObject *)newCSRMatObject( L->dim, nnzl );
    if( Lcsr != NULL ) {
        coord2csr( self->data->n, nnzl, il, jl, l,
                   Lcsr->ind, Lcsr->col, Lcsr->val );
    }
    */

    //free( inv_perm );
    NLPy_Free( d );
    NLPy_Free( id );
    NLPy_Free( jd );
    NLPy_Free( l );
    NLPy_Free( il );
    NLPy_Free( jl );
    NLPy_Free( colrhs );

    self->data->fetched = 1;

    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static char Pyma27_ma27_Doc[] = "Solve Ax=b using a direct multifrontal method";

static PyObject *Pyma27_ma27( Pyma27Object *self, PyObject *args ) {

    PyArrayObject *a_x, *a_rhs, *a_res;
    double        *x, *rhs;
    int            i, j, k;
    int            error, comp_resid;

    /* We read a right-hand side and a solution */
    if( !PyArg_ParseTuple( args,
                           "O!O!O!i:ma27",
                           &PyArray_Type, &a_rhs,
                           &PyArray_Type, &a_x,
                           &PyArray_Type, &a_res, &comp_resid ) ) return NULL;

    if( a_rhs->descr->type_num != NPY_DOUBLE ) return NULL;
    if( a_x->descr->type_num != NPY_DOUBLE ) return NULL;
    if( a_res->descr->type_num != NPY_DOUBLE ) return NULL;
    if( !a_rhs ) return NULL;                          /* conversion error */
    if( a_rhs->nd != 1 ) return NULL;           /* b must have 1 dimension */
    if( a_rhs->dimensions[0] != self->data->n ) return NULL; /* and size n */
    if( !a_x ) return NULL;
    if( a_x->nd != 1 ) return NULL;
    if( a_x->dimensions[0] != self->data->n ) return NULL;
    if( !a_res ) return NULL;
    if( a_res->nd != 1 ) return NULL;
    if( a_res->dimensions[0] != self->data->n ) return NULL;

    rhs = (double *)a_rhs->data;
    x = (double *)a_x->data;

    /* Copy rhs into x; it will be overwritten by Ma27_Solve() */
    cblas_dcopy( self->data->n, rhs, 1, x, 1 );

    /* Solve */
    error = Ma27_Solve( self->data, x );
    if( error ) {
        fprintf( stderr, " Error return code from Solve: %-d\n", error );
        return NULL;
    }

    /* Compute residual r = rhs - Ax */
    if( comp_resid ) {
        self->data->residual = (double *)a_res->data;
        cblas_dcopy( self->data->n, rhs, 1, self->data->residual, 1 );
        for( k = 0; k < self->data->nz; k++ ) {
            i = self->data->irn[k] - 1;  /* Fortran indexing */
            j = self->data->icn[k] - 1;
            self->data->residual[i] -= self->a[k] * x[j];
            if( i != j )
                self->data->residual[j] -= self->a[k] * x[i];
        }
    }

    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static char Pyma27_Stats_Doc[] = "Obtain statistics on factorization";

static PyObject *Pyma27_Stats( Pyma27Object *self, PyObject *args ) {

    /* Return statistics on the solve */
    /* info[8]  = # real    words used during the factorization,
     * info[9]  = # integer   "    "      "    "       "
     * info[10] = # data    compresses performed during the analysis,
     * info[11] = # real      "           "        "     "     "
     * info[12] = # integer   "           "        "     "     "
     * info[13] = # 2x2 pivots used
     * info[14] = # negative eigenvalues detected.
     */

    return Py_BuildValue( "iiiiiiii", self->data->info[8],
                                      self->data->info[9],
                                      self->data->info[10],
                                      self->data->info[11],
                                      self->data->info[12],
                                      self->data->info[13],
                                      self->data->info[14],
                                      self->data->rank      );
}

/* ========================================================================== */

static char Pyma27_refine_Doc[] = "Perform iterative refinements";

static PyObject *Pyma27_refine( Pyma27Object *self, PyObject *args ) {

    PyArrayObject *a_x, *a_rhs, *a_res;
    double        *x, *rhs;
    double         tol;
    int            nitref, nerror;

    /* We read the number of iterative refinements, x and a tolerance */
    if( !PyArg_ParseTuple( args, "O!O!O!di:refine", &PyArray_Type, &a_x,
                                                    &PyArray_Type, &a_res,
                                                    &PyArray_Type, &a_rhs,
                                                    &tol, &nitref ) )
        return NULL;
    if( a_x->descr->type_num != NPY_DOUBLE ) return NULL;
    if( !a_x ) return NULL;
    if( a_x->nd != 1 ) return NULL;
    if( a_x->dimensions[0] != self->data->n ) return NULL;
    if( a_rhs->descr->type_num != NPY_DOUBLE ) return NULL;
    if( !a_rhs ) return NULL;
    if( a_rhs->nd != 1 ) return NULL;
    if( a_rhs->dimensions[0] != self->data->n ) return NULL;
    if( a_res->descr->type_num != NPY_DOUBLE ) return NULL;
    if( !a_res ) return NULL;
    if( a_res->nd != 1 ) return NULL;
    if( a_res->dimensions[0] != self->data->n ) return NULL;

    x = (double *)a_x->data;
    rhs = (double *)a_rhs->data;
    self->data->residual = (double *)a_res->data;

    nerror = Ma27_Refine( self->data, x, rhs, self->a, tol, nitref );
    if( nerror == -10 ) return NULL;
    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

/* This is necessary as Pyma27_ma27 takes a Pyma27Object* as argument */

static PyMethodDef Pyma27_special_methods[] = {
  { "ma27",      (PyCFunction)Pyma27_ma27,
    METH_VARARGS, Pyma27_ma27_Doc       },
  { "fetchperm", (PyCFunction)Pyma27_fetch_perm,
    METH_VARARGS, Pyma27_fetch_perm_Doc },
  { "fetchlb",   (PyCFunction)Pyma27_fetch_lb,
    METH_VARARGS, Pyma27_fetch_lb_Doc   },
  { "stats",     (PyCFunction)Pyma27_Stats,
    METH_VARARGS, Pyma27_Stats_Doc      },
  { "refine",    (PyCFunction)Pyma27_refine,
    METH_VARARGS, Pyma27_refine_Doc     },
  { NULL,        NULL,
    0,            NULL                  }
};

/* ========================================================================== */

static char Pyma27_factor_Doc[] = "Factorize input matrix";

static PyObject *Pyma27_factor( PyObject *self, PyObject *args ) {

    /* Input must be the lower triangle of a symmetric matrix */

    Pyma27Object  *rv;                    /* Return value */
    PyObject      *mat;                   /* Input matrix */
    PyObject      *sqd;                   /* SQD matrix flag */

    /* Read input matrix and limited memory factor */
    if( !PyArg_ParseTuple( args, "OO:factor", &mat, &sqd ) )
        return NULL;

    /* Spawn new Pyma27 Object, containing matrix factors */
    rv = NewPyma27Object( (LLMatObject *)mat, sqd );
    if( rv == NULL ) return NULL;

    return (PyObject *)rv;
}

/* ========================================================================== */

/*
 *     D e f i n i t i o n   o f   P y m a 2 7   c o n t e x t   t y p e
 */

/* ========================================================================== */

static PyTypeObject Pyma27Type = {
    PyObject_HEAD_INIT(NULL)
    0,
    "pyma27_context",
    sizeof(Pyma27Object),
    0,
    (destructor)Pyma27_dealloc,  /* tp_dealloc */
    0,                           /* tp_print */
    (getattrfunc)Pyma27_getattr, /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    0,                           /* tp_repr */
    0,                           /* tp_as_number*/
    0,                           /* tp_as_sequence*/
    0,                           /* tp_as_mapping*/
    0,                           /* tp_hash */
    0,                           /* tp_call*/
    0,                           /* tp_str*/
    0,                           /* tp_getattro*/
    0,                           /* tp_setattro*/
    0,                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,          /* tp_flags*/
    "PyMa27 Context Object",     /* tp_doc */
};

/* ========================================================================== */

/*
 *            D e f i n i t i o n   o f   P y m a 2 7   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef Pyma27Methods[] = {
    { "factor",  Pyma27_factor,  METH_VARARGS, Pyma27_factor_Doc  },
    { NULL,     NULL,            0,            NULL               }
};

/* ========================================================================== */

static void Pyma27_dealloc( Pyma27Object *self ) {

    NLPy_Free( self->a );
    Ma27_Finalize( self->data );
    PyObject_Del(self);
}

/* ========================================================================== */

static PyObject *Pyma27_getattr( Pyma27Object *self, char *name ) {

    if( strcmp( name, "shape" ) == 0 )
    return Py_BuildValue( "(i,i)", self->data->n, self->data->n );
    if( strcmp( name, "nnz" ) == 0 )
    return Py_BuildValue( "i", self->data->nz );
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
    return Py_FindMethod( Pyma27_special_methods, (PyObject *)self, name );
}

/* ========================================================================== */

DL_EXPORT( void ) init_pyma27( void ) {

    PyObject *m, *d;

    //Pyma27Type.ob_type = &PyType_Type;
    if( PyType_Ready( &Pyma27Type ) < 0 ) return;

    m = Py_InitModule3( "_pyma27", Pyma27Methods, "Python interface to MA27" );

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "Pyma27Type", (PyObject *)&Pyma27Type);

    import_array( );         /* Initialize the Numarray module. */
    import_spmatrix( );      /* Initialize the PySparse module. */

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("Unable to initialize module pyma27");

    return;
}

/* ========================================================================== */

void coord2csr( int n, int nz, int *irow, int *jcol, double *val,
                int *iptr, int *jind, double *xval ) {

    int i, start, elem, rowcnt;

    for( i = 0; i < n+1; i++ ) iptr[i] = 0;
    for( i = 0; i < nz;  i++ ) iptr[ irow[i] ]++;  /* length of each row */
    /* Obtain starting index of each row */
    start = 0;
    for( i = 0; i < n+1; i++ ) {
        rowcnt = iptr[i];
        iptr[i] = start;
        start += rowcnt;
    }
    /* Fill in matrix */
    for( elem = 0; elem < nz; elem++ ) {
        i = irow[elem];
        start = iptr[i];
        xval[start] = val[elem];
        jind[start] = jcol[elem];
        iptr[i] = start+1;
    }
    /* Restore iptr */
    for( i = nz; i >= 0; i-- )
        iptr[i+1] = iptr[i];
    iptr[0] = 0;
    return;
}

/* ========================================================================== */

/*
 *                    E n d   o f   m o d u l e   P y m a 2 7
 */

/* ========================================================================== */
