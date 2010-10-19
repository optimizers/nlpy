
/* ========================================================================== */

/*
 *                                   P y m a 5 7
 *
 *   P y t h o n   i n t e r f a c e   t o   t h e   M A 5 7   p a c k a g e
 */

/* ========================================================================== */

/* 
   $Revision:$
   $Date:$
*/

/* Indicate that module is split across several files */
#define PY_ARRAY_UNIQUE_SYMBOL ma57

#include <Python.h>                /* Main Python header file */
#include <cblas.h>
#include "numpy/arrayobject.h"     /* NumPy header */
#include "spmatrix.h"
#include "spmatrix_api.h"
#include "ll_mat.h"
#include "ma57.h"

#define  SYMMETRIC 1  /* for SpMatrix_NewLLMatObject() */

/* ========================================================================== */

/*
 *    D e f i n i t i o n   o f   P y m a 5 7   c o n t e x t   o b j e c t
 */

/* ========================================================================== */

staticforward PyTypeObject Pyma57Type;    /* Precise definition appears below */

typedef struct Pyma57Object {
  PyObject_VAR_HEAD
  Ma57_Data  *data;
  double     *a;
} Pyma57Object;

#define Pyma57Object_Check(v)  ((v)->ob_type == &Pyma57Type)

/* ========================================================================== */

DL_EXPORT( void ) init_pyma57( void );
static PyObject     *Pyma57_ma57(       Pyma57Object *self,  PyObject *args );
static PyObject     *Pyma57_factorize(  Pyma57Object *self,  PyObject *args );
static PyObject     *Pyma57_refine(     Pyma57Object *self,  PyObject *args );
static PyObject     *Pyma57_analyze(    PyObject     *self,  PyObject *args );
static void          Pyma57_dealloc(    Pyma57Object *self                  );
static PyObject     *Pyma57_getattr(    Pyma57Object *self,  char *name     );
static PyObject     *Pyma57_Stats(      Pyma57Object *self,  PyObject *args );
static PyObject     *Pyma57_fetch_perm( Pyma57Object *self                  );
//static PyObject     *Pyma57_fetch_lb(   Pyma57Object *self,  PyObject *args );
static PyObject *NewPyma57Object(   LLMatObject  *llmat, PyObject *sqd  );
extern PyObject *newCSRMatObject(int dim[], int nnz);
void coord2csr( int n, int nz, int *irow, int *jcol, double *val,
                int *iptr, int *jind, double *xval );

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */


static PyObject *NewPyma57Object( LLMatObject *llmat, PyObject *sqd ) {

  Pyma57Object *self;
  int           n  = llmat->dim[0],
                nz = llmat->nnz;
  int           i, k, elem;
  int           error;

  /* Create new instance of object */
  if( !(self = PyObject_New( Pyma57Object, &Pyma57Type ) ) ) {
    fprintf(stderr, "Cannot allocate memory for Pyma57Type object.\n");
    return PyErr_NoMemory( ); // NULL;
  }

  self->data = Ma57_Initialize( nz, n, NULL );

  /* Set pivot-for-stability threshold is matrix is SQD */
  if( sqd == Py_True ) {
    self->data->cntl[0] = 1.0e-15;
    self->data->cntl[1] = 1.0e-15;
    self->data->icntl[6] = 1;
    //self->data->icntl[6] = 3; // No pivoting, error if |pivot| < cntl[1]
  }

  /* Get matrix in coordinate format. Adjust indices to be 1-based. */
  elem = 0;
  for( i = 0; i < n; i++ ) {
    k = llmat->root[i];
    while( k != -1 ) {
      self->data->irn[ elem ] = i + 1;
      self->data->jcn[ elem ] = llmat->col[k] + 1;
      //self->a[ elem ] = llmat->val[k];
      k = llmat->link[k];
      elem++;
    }
  }

  /* Analyze */
  error = Ma57_Analyze( self->data );
  if( error ) {
    fprintf( stderr, " Error return code from Analyze: %-d\n", error );
    return NULL;
  }
  return (PyObject *)self;
}

/* ========================================================================== */

static char Pyma57_factorize_Doc[] = "Factorize matrix";

static PyObject *Pyma57_factorize( Pyma57Object *self, PyObject *args ) {

  PyObject    *mat;
  LLMatObject *llmat;
  int          n, nz;               
  int          i, k, elem;
  int          error;

  /* See if input matrix has changed since analyze phase */
  if( !PyArg_ParseTuple( args, "O:factorize", &mat ) ) return NULL;

  llmat = (LLMatObject *)mat;
  n  = llmat->dim[0];
  nz = llmat->nnz;

  /* Make temporary copy of matrix. Is there a way to avoid this?  */
  self->a = (double *)NLPy_Calloc( nz, sizeof(double) );
  if( self->a == NULL ) return PyErr_NoMemory();

  elem = 0;
  for( i = 0; i < n; i++ ) {
    k = llmat->root[i];
    while( k != -1 ) {
      self->a[ elem ] = llmat->val[k];
      k = llmat->link[k];
      elem++;
    }
  }

  /* Factorize */
  error = Ma57_Factorize( self->data, self->a );
  if( error ) {
    fprintf( stderr, " Error return code from Factorize: %-d\n", error );
    return NULL;
  }  

  /* Find out if matrix was rank deficient */
  self->data->rank = self->data->info[24];
  self->data->rankdef = (self->data->rank < self->data->n) ? 1 : 0;

  Py_INCREF( Py_None );
  return Py_None;
}

/* ========================================================================== */

static char Pyma57_fetch_perm_Doc[] = "Fetch variables permutation computed by MA57";

static PyObject *Pyma57_fetch_perm( Pyma57Object *self ) {

  PyObject *perm;
  int       i, err;

  /* New list to hold the pivot order */
  perm = PyList_New( 0 );
  if( !perm ) return NULL;
  for( i = 0; i < self->data->n; i++ ) {
    err = PyList_Append( perm,
                         (PyObject *)PyInt_FromLong(self->data->keep[i]));
    if( err < 0 ) {
      Py_DECREF( perm );
      return NULL;
    }
  }

  return Py_BuildValue( "O", perm );
}

/* ========================================================================== */

/* static char Pyma27_fetch_lb_Doc[] = "Fetch factors of A computed by MA27"; */

/* static PyObject *Pyma27_fetch_lb( Pyma27Object *self, PyObject *args ) { */

/*   PyObject     *Lmat, *Dmat; */
/*   LLMatObject  *L, *D; */
/*   double       *l, *d; */
/*   int          *id, *jd, *il, *jl;  /\* L and D^{-1} in coordinate format *\/ */
/*   int           nnzL, nnzD, nblk, n1x1, n2x2, latop, liwm1, *iwp1, *colrhs; */
/*   int           i, ii, jj; */
/*   int           n = self->data->n; */
/*   int *inv_perm; */

/*   if( !PyArg_ParseTuple( args, "OO", &Lmat, &Dmat ) ) return NULL; */
/*   L = (LLMatObject *)Lmat; */
/*   D = (LLMatObject *)Dmat; */
    
/*   /\* Obtain the number of nonzeros in factors L and D^{-1} *\/ */
/*   nblk = abs( self->data->iw[0] );   /\* # block pivots *\/ */
/*   if( nblk == 0 ) { */
/*     Py_INCREF( Py_None ); */
/*     return Py_None; */
/*   } */
/*   iwp1 = (self->data->iw)+1; */
/*   liwm1 = self->data->liw - 1; */
/*   MA27QDEMASC( &n, iwp1, &liwm1, self->data->iw1, */
/*                &nblk, &latop, self->data->icntl ); */
    
/*   n2x2 = self->data->info[13];   /\* No. of 2x2 pivots. *\/ */
/*   n1x1 = n - 2*n2x2;             /\* No. of 1x1 pivots. *\/ */
/*   nnzD = n1x1 + 4*n2x2;          /\* 1 nz for each 1x1, 4 nz's for each 2x2. *\/ */
/*   nnzL = n + latop;              /\* An upper bound only, not exact. *\/ */

/*   /\* Allocate space for D^{-1} and L *\/ */
/*   d  = (double *)NLPy_Calloc( nnzD, sizeof(double) ); */
/*   l  = (double *)NLPy_Calloc( nnzL, sizeof(double) ); */
/*   id = (int *)NLPy_Calloc( nnzD, sizeof(int) ); */
/*   jd = (int *)NLPy_Calloc( nnzD, sizeof(int) ); */
/*   il = (int *)NLPy_Calloc( nnzL, sizeof(int) ); */
/*   jl = (int *)NLPy_Calloc( nnzL, sizeof(int) ); */

/*   colrhs = (int *)NLPy_Calloc( self->data->maxfrt, sizeof(int) ); */

/*   /\* Obtain lower triangle of D^{-1} and upper triangle of L *\/ */
/*   MA27FACTORS( &n, self->data->factors, &(self->data->la), */
/*                iwp1, &liwm1, &(self->data->maxfrt), self->data->iw1, */
/*                &nblk, &latop, self->data->icntl, colrhs, */
/*                &nnzD, id, jd, d, */
/*                &nnzL, il, jl, l ); */
    
/*   /\* At this point, nnzL is exact.  Build sparse matrices D^{-1} */
/*    * and L.  Account for 0-based indexing *\/ */
/*   for( i = 0; i < nnzL; i++ ) */
/*     SpMatrix_LLMatSetItem( L, jl[i]-1, il[i]-1, l[i] ); */
    
/*   for( i = 0; i < nnzD; i++ ) */
/*     SpMatrix_LLMatSetItem( D, id[i]-1, jd[i]-1, d[i] ); */

/*   NLPy_Free( d ); */
/*   NLPy_Free( id ); */
/*   NLPy_Free( jd ); */
/*   NLPy_Free( l ); */
/*   NLPy_Free( il ); */
/*   NLPy_Free( jl ); */
/*   NLPy_Free( colrhs ); */

/*   self->data->fetched = 1; */

/*   Py_INCREF( Py_None ); */
/*   return Py_None; */
/* } */

/* ========================================================================== */

static char Pyma57_ma57_Doc[] = "Solve Ax=b using a direct multifrontal method";

static PyObject *Pyma57_ma57( Pyma57Object *self, PyObject *args ) {

  PyArrayObject *a_x, *a_rhs, *a_res;
  PyObject      *get_resid;
  double        *x, *rhs;
  int            error;

  /* We read a right-hand side and a solution */
  if( !PyArg_ParseTuple( args,
                         "O!O!O!O:ma57",
                         &PyArray_Type, &a_rhs,
                         &PyArray_Type, &a_x,
                         &PyArray_Type, &a_res, &get_resid ) ) return NULL;

  if( a_rhs->descr->type_num != NPY_DOUBLE ) return NULL;
  if( a_x->descr->type_num != NPY_DOUBLE ) return NULL;
  if( a_res->descr->type_num != NPY_DOUBLE ) return NULL;
  if( !a_rhs ) return NULL;                               /* conversion error */
  if( a_rhs->nd != 1 ) return NULL;                  /* must have 1 dimension */
  if( a_rhs->dimensions[0] != self->data->n ) return NULL;      /* and size n */
  if( !a_x ) return NULL;
  if( a_x->nd != 1 ) return NULL;
  if( a_x->dimensions[0] != self->data->n ) return NULL;
  if( !a_res ) return NULL;
  if( a_res->nd != 1 ) return NULL;
  if( a_res->dimensions[0] != self->data->n ) return NULL;

  rhs = (double *)a_rhs->data;
  x = (double *)a_x->data;
  self->data->residual = (double *)a_res->data;

  if( get_resid == Py_True )  /* Solve and compute residual r = rhs - Ax */
    error = Ma57_Refine( self->data, x, rhs, self->a, 1, 0 );
  else {            /* Just solve */
    cblas_dcopy( self->data->n, rhs, 1, x, 1 ); // x<- rhs ; will be overwritten
    error = Ma57_Solve( self->data, x );
  }

  if( error ) {
    fprintf( stderr, " Error return code from Solve: %-d\n", error );
    return NULL;
  }

  Py_INCREF( Py_None );
  return Py_None;
}

/* ========================================================================== */

static char Pyma57_Stats_Doc[] = "Obtain statistics on factorization";

static PyObject *Pyma57_Stats( Pyma57Object *self, PyObject *args ) {

  /* Return statistics on the solve */
  /* info[13] = number of entries in factors,
   * info[14] = storage for real data of factors,
   * info[15] = storage for int  data of factors,
   * info[20] = largest front size,
   * info[21] = number of 2x2 pivots,
   * info[23] = number of negative eigenvalues,
   * info[24] = matrix rank
   */

  return Py_BuildValue( "iiiiiii",
                        self->data->info[13],
                        self->data->info[14],
                        self->data->info[15],
                        self->data->info[20],
                        self->data->info[21],
                        self->data->info[23],
                        self->data->info[24] );
}

/* ========================================================================== */

static char Pyma57_refine_Doc[] = "Perform iterative refinements";

static PyObject *Pyma57_refine( Pyma57Object *self, PyObject *args ) {

  PyArrayObject *a_x, *a_rhs, *a_res;
  double        *x, *rhs;
  int            nitref, nerror;

  /* We read the number of iterative refinements, x and a rhs */
  if( !PyArg_ParseTuple( args, "O!O!O!i:refine",
                         &PyArray_Type, &a_x,
                         &PyArray_Type, &a_res,
                         &PyArray_Type, &a_rhs,
                         &nitref ) )
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

  nerror = Ma57_Refine( self->data, x, rhs, self->a, nitref, 2 );
  if( nerror == -10 ) return NULL;
  return Py_BuildValue("dddddddd",
                       self->data->rinfo[10],    // 1st cond number estimate
                       self->data->rinfo[11],    // 2nd cond number estimate
                       self->data->rinfo[5],     // 1st backward err estimate
                       self->data->rinfo[6],     // 2nd backward err estimate
                       self->data->rinfo[12],    // direct error estimate
                       self->data->rinfo[7],     // Inf-norm of input matrix
                       self->data->rinfo[8],     // Inf-norm of solution
                       self->data->rinfo[9]);    // Relative residual
}

/* ========================================================================== */

/* This is necessary as Pyma57_ma57 takes a Pyma57Object* as argument */

static PyMethodDef Pyma57_special_methods[] = {
  { "ma57",      (PyCFunction)Pyma57_ma57,
    METH_VARARGS, Pyma57_ma57_Doc                 },
  { "factorize", (PyCFunction)Pyma57_factorize,
    METH_VARARGS, Pyma57_factorize_Doc            },
  { "fetchperm", (PyCFunction)Pyma57_fetch_perm,
    METH_VARARGS, Pyma57_fetch_perm_Doc           },
  //{ "fetchlb",   (PyCFunction)Pyma57_fetch_lb,
  //  METH_VARARGS, Pyma57_fetch_lb_Doc   },
  { "stats",     (PyCFunction)Pyma57_Stats,
    METH_VARARGS, Pyma57_Stats_Doc                },
  { "refine",    (PyCFunction)Pyma57_refine,
    METH_VARARGS, Pyma57_refine_Doc               },
  { NULL,         NULL,
    0,            NULL                            }
};

/* ========================================================================== */

static char Pyma57_analyze_Doc[] = "Analyze input matrix";

static PyObject *Pyma57_analyze( PyObject *self, PyObject *args ) {

  /* Input must be the lower triangle of a symmetric matrix */

  PyObject  *rv;                    /* Return value */
  PyObject  *mat;                   /* Input matrix */
  PyObject  *sqd;                   /* SQD matrix flag */

  /* Read input matrix and limited memory factor */
  if( !PyArg_ParseTuple( args, "OO:factor", &mat, &sqd ) )
    return NULL;

  /* Spawn new Pyma57 Object, containing matrix symbolic factors */
  rv = NewPyma57Object( (LLMatObject *)mat, sqd );
  if( rv == NULL ) return NULL;

  return rv;
}

/* ========================================================================== */

/*
 *     D e f i n i t i o n   o f   P y m a 5 7   c o n t e x t   t y p e
 */

/* ========================================================================== */

static PyTypeObject Pyma57Type = {
  PyObject_HEAD_INIT(NULL)
  0,
  "pyma57_context",
  sizeof(Pyma57Object),
  0,
  (destructor)Pyma57_dealloc,  /* tp_dealloc */
  0,                           /* tp_print */
  (getattrfunc)Pyma57_getattr, /* tp_getattr */
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
  "PyMa57 Context Object",     /* tp_doc */
};

/* ========================================================================== */

/*
 *            D e f i n i t i o n   o f   P y m a 5 7   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef Pyma57Methods[] = {
  { "analyze",  Pyma57_analyze,  METH_VARARGS, Pyma57_analyze_Doc },
  { NULL,       NULL,            0,            NULL               }
};

/* ========================================================================== */

static void Pyma57_dealloc( Pyma57Object *self ) {

  NLPy_Free( self->a );
  Ma57_Finalize( self->data );
  PyObject_Del(self);
}

/* ========================================================================== */

static PyObject *Pyma57_getattr( Pyma57Object *self, char *name ) {

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
  return Py_FindMethod( Pyma57_special_methods, (PyObject *)self, name );
}

/* ========================================================================== */

DL_EXPORT( void ) init_pyma57( void ) {

  PyObject *m, *d;

  if( PyType_Ready( &Pyma57Type ) < 0 ) return;

  m = Py_InitModule3( "_pyma57", Pyma57Methods, "Python interface to MA57" );

  d = PyModule_GetDict(m);
  PyDict_SetItemString(d, "Pyma57Type", (PyObject *)&Pyma57Type);

  import_array( );         /* Initialize the NumPy module.    */
  import_spmatrix( );      /* Initialize the PySparse module. */

  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("Unable to initialize module pyma57");

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
 *                    E n d   o f   m o d u l e   P y m a 5 7
 */

/* ========================================================================== */
