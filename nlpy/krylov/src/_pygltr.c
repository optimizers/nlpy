
/* ========================================================================== */

/*
 *                                  P y g l t r
 *
 *   P y t h o n   i n t e r f a c e   t o   t h e   G L T R   p a c k a g e
 */

/* ========================================================================== */

/* Indicate that module is split across several files */
#define PY_ARRAY_UNIQUE_SYMBOL pygltr

#include "Python.h"                /* Main Python header file */
#include "arrayflavor.h"
#include "pysparse/spmatrix.h"
#include "pysparse/spmatrix_api.h"
#include "pysparse/ll_mat.h"

/* ========================================================================== */

/*
 *    D e f i n i t i o n   o f   P y g l t r   c o n t e x t   o b j e c t
 */

/* ========================================================================== */

staticforward PyTypeObject PygltrType;    /* Precise definition appears below */

typedef struct PygltrObject {
    PyObject_VAR_HEAD
    int n;                   /* Current problem dimension      */
    int niter;               /* Current iteration count        */
    int nc;                  /* Negative curvature flag        */
    int ierr;                /* Return code from GLTR          */
    double *g;               /* Initial gradient vector        */
    double *r;               /* Current gradient vector        */
    double *step;            /* Current step                   */
    double *vector;          /* Temporary vector               */
    double  f;               /* Current function value         */
    double  lambda;          /* Current multiplier             */
    double  snorm;           /* Norm of current step           */
    double  radius;          /* Current trust-region radius    */
    double  stop_relative;   /* Relative stopping tolerance    */
    double  stop_absolute;   /* Absolute stopping tolerances   */
    int     itmax;           /* Maximum number of iterations   */
    int     litmax;          /* Max number of Lanczos iters    */
    int     unitm;           /* Flag to represent prec = id    */
    int     ST;              /* Flag for Steihaug-Toint        */
    int     boundary;        /* Soln thought to be on boundary */
    int     equality;        /* Soln must be on boundary       */
    double  fraction_opt;    /* Acceptable fract of optimality */
} PygltrObject;

/* ========================================================================== */

/* Support different Fortran compilers */
#ifdef _AIX
#define FUNDERSCORE(a) a
#else
#define FUNDERSCORE(a) a##_
#endif

#define PYGLTR    FUNDERSCORE(pygltr)

#define ZERO     (double)0.0
#define PYGLTR_MAX(a,b) ((a) > (b)) ? (a) : (b)

DL_EXPORT( void ) init_pygltr( void );
static PygltrObject *NewPygltrObject( PyObject *args );
static PyObject     *Pygltr_solve( PygltrObject *self, PyObject *args );
static PyObject     *Pygltr_gltr( PyObject *self, PyObject *args );
static PyObject     *Pygltr_getattr( PygltrObject *self, char *name );
static void          Pygltr_dealloc( PygltrObject *self );

extern void PYGLTR( int *n, double *f, double *g, double *vector,
		            double *radius, double *stop_relative,
                    double *stop_absolute,
                    int *itmax, int *litmax, int *unitm, int *ST,
                    int *boundary, int *equality, double *fraction_opt,
		            double *step, double *multiplier, double *snorm,
                    int *niter, int *nc, int *ierr, int *initial );

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */

static PygltrObject *NewPygltrObject( PyObject *args ) {

    PygltrObject  *self;
    PyArrayObject *a_g, *a_step, *a_vector;
    double        *pg;
    double         radius, stop_rel, stop_abs, fraction_opt;
    int            n, itmax, litmax, unitm, ST, boundary, equality;
    int            initial = 1;   /* Marks initial call to GLTR */
    int            i;

    /* Parse arguments */
    if( !PyArg_ParseTuple( args, "O!O!O!dddiiiiiid",
                                 &PyArray_Type, &a_g,
                                 &PyArray_Type, &a_step,
                                 &PyArray_Type, &a_vector,
                                 &radius, &stop_rel, &stop_abs,
                                 &itmax, &litmax, &unitm, &ST,
                                 &boundary, &equality, &fraction_opt ) )
        return NULL;

    if( radius <= 0 ) return NULL;

    /* Make sure arrays are 1-dimensional arrays of doubles of length n */
    if( !a_g ) return NULL;
    if( a_g->nd != 1 ) return NULL;
    if( a_g->descr->type_num != tFloat64 ) return NULL;
    n = a_g->dimensions[0];
    if( n <= 0 ) return NULL;
    
    if( !a_step ) return NULL;
    if( a_step->nd != 1 ) return NULL;
    if( a_step->dimensions[0] != n ) return NULL;
    if( a_step->descr->type_num != tFloat64 ) return NULL;

    if( !a_vector ) return NULL;
    if( a_vector->nd != 1 ) return NULL;
    if( a_vector->dimensions[0] != n ) return NULL;
    if( a_vector->descr->type_num != tFloat64 ) return NULL;
    
    /* Create new instance of object */
    if( !(self = PyObject_New( PygltrObject, &PygltrType ) ) )
        return NULL;
    
    /* Allocate room for problem gradient */
    self->g = (double *)malloc( n * sizeof( double ) );
    if( self->g == NULL ) return NULL;

    /* Allocate room for r */
    self->r = (double *)malloc( n * sizeof( double ) );
    if( self->r == NULL ) return NULL;

    /* Store stopping tolerances */
    self->stop_relative = stop_rel; /* PYGLTR_MAX( ZERO, stop_rel ); */
    self->stop_absolute = stop_abs; /* PYGLTR_MAX( ZERO, stop_abs ); */

    /* Associate data->step */
    self->step = (double *)a_step->data;

    /* Associate data->vector */
    self->vector = (double *)a_vector->data;

    /* Initialize gradient */
    pg = (double *)a_g->data;
    for( i = 0; i < n; i++ ) {
        self->g[i]      = pg[i];
        self->r[i]      = pg[i];
    }

    /* Store problem data */
    self->n            = n;
    self->radius       = radius;
    self->itmax        = itmax;
    self->litmax       = litmax;
    self->unitm        = unitm;
    self->ST           = ST;
    self->boundary     = boundary;
    self->equality     = equality;
    self->fraction_opt = fraction_opt;

    /* Set remaining components of object to default values */
    self->f      = ZERO;
    self->lambda = ZERO;
    self->snorm  = ZERO;
    self->niter  = 0;
    self->nc     = 0;
    self->ierr   = 0;

    /* Initialize internal GLTR structures */
    /* Since initial = 1, this includes a call to GLTR_initialize() */
    PYGLTR( &(self->n), &(self->f), self->r, self->vector, &(self->radius),
            &(self->stop_relative), &(self->stop_absolute),
            &(self->itmax), &(self->litmax), &(self->unitm), &(self->ST),
            &(self->boundary), &(self->equality), &(self->fraction_opt),
            self->step, &(self->lambda), &(self->snorm),  &(self->niter),
            &(self->nc), &(self->ierr), &initial );

    return self;
}

/* ========================================================================== */

static char Pygltr_reassign_Doc[] = "Assign updated value to VECTOR component";

static PyObject *Pygltr_reassign( PygltrObject *self, PyObject *args ) {

    PyArrayObject *a_vector;
    double *vp;
    int i;

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_vector ) )
        return NULL;
    if( a_vector->nd != 1 ) return NULL;
    if( a_vector->dimensions[0] != self->n ) return NULL;
    if( a_vector->descr->type_num != tFloat64 ) return NULL;

    vp = (double *)a_vector->data;
    for( i = 0; i < self->n; i++ )
        self->vector[i] = vp[i];

    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static char Pygltr_solve_Doc[] = "Solve trust-region subproblem iteratively";

static PyObject *Pygltr_solve( PygltrObject *self, PyObject *args ) {

    PyArrayObject *a_step, *a_vector;
    int i, initial = 0, exit_loop;
    
    if( self == NULL ) return NULL;

    /* Read two vectors of length n */
    if( !PyArg_ParseTuple( args, "O!O!",
                                 &PyArray_Type, &a_step,
                                 &PyArray_Type, &a_vector ) )
        return NULL;

    self->step = (double *)a_step->data;
    self->vector = (double *)a_vector->data;
    exit_loop = 0;

    /* Call solver --- gradient reinitialization will be handled directly */
    while( !exit_loop ) {
        PYGLTR( &(self->n), &(self->f), self->r, self->vector, &(self->radius),
                &(self->stop_relative), &(self->stop_absolute),
                &(self->itmax), &(self->litmax), &(self->unitm), &(self->ST),
                &(self->boundary), &(self->equality), &(self->fraction_opt),
	            self->step, &(self->lambda), &(self->snorm),
                &(self->niter), &(self->nc), &(self->ierr), &initial );

        /* See if gradient must be re-initialized */
        if( self->ierr == 5 ) {
            for( i = 0; i < self->n; i++ )
                self->r[i] = self->g[i];
        } else {
            exit_loop = 1;
        }
    }

    /* Return scalar values */
    return Py_BuildValue( "dddiii", self->f, self->lambda, self->snorm,
                                     self->niter, self->nc, self->ierr );
}

/* ========================================================================== */

/* This is necessary as Pygltr_gltr takes a PygltrObject* as argument */

static PyMethodDef Pygltr_special_methods[] = {
    { "solve",    (PyCFunction)Pygltr_solve,    METH_VARARGS, Pygltr_solve_Doc    },
    { "reassign", (PyCFunction)Pygltr_reassign, METH_VARARGS, Pygltr_reassign_Doc },
    { NULL,    NULL,                      0,            NULL                      }
};

/* ========================================================================== */

static char Pygltr_gltr_Doc[] = "Initialize new PyGltrObject";

static PyObject *Pygltr_gltr( PyObject *self, PyObject *args ) {

    PygltrObject *rv;
    
    /* Pass args directly over to NewPygltrObject() */
    rv = NewPygltrObject( args );
    if( rv == NULL ) return NULL;
    return (PyObject *)rv;
}

/* ========================================================================== */

/*
 *       D e f i n i t i o n   o f   P y g l t r   c o n t e x t   t y p e
 */

/* ========================================================================== */

static PyTypeObject PygltrType = {
    PyObject_HEAD_INIT(NULL)
    0,
    "pygltr_context",
    sizeof(PygltrObject),
    0,
    (destructor)Pygltr_dealloc,  /* tp_dealloc */
    0,                           /* tp_print */
    (getattrfunc)Pygltr_getattr, /* tp_getattr */
    0,			 	 /* tp_setattr */
    0,				 /* tp_compare */
    0,				 /* tp_repr */
    0,				 /* tp_as_number*/
    0,				 /* tp_as_sequence*/
    0,				 /* tp_as_mapping*/
    0,				 /* tp_hash */
};

/* ========================================================================== */

/*
 *            D e f i n i t i o n   o f   P y g l t r   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef PygltrMethods[] = {
    { "gltr",  Pygltr_gltr, METH_VARARGS, Pygltr_gltr_Doc },
    { NULL,    NULL,        0,            NULL            }       /* Sentinel */
};


/* ========================================================================== */

static void Pygltr_dealloc( PygltrObject *self ) {

    if( self->g ) free( self->g );
    if( self->r ) free( self->r );
    PyObject_Del(self);
}

/* ========================================================================== */

static PyObject *Pygltr_getattr( PygltrObject *self, char *name ) {

    if( strcmp( name, "dim" ) == 0 )
	    return Py_BuildValue( "i", self->n );
    if( strcmp( name, "__members__" ) == 0 ) {
	    char *members[] = {"dim"};
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
    return Py_FindMethod( Pygltr_special_methods, (PyObject *)self, name );
}

/* ========================================================================== */

DL_EXPORT( void ) init_pygltr( void ) {

    (void)Py_InitModule3( "_pygltr", PygltrMethods, "Python interface to GLTR" );

    import_array( );         /* Initialize the Numeric/Numarray module. */

    /* Check for errors */
    if (PyErr_Occurred())
	    Py_FatalError("Unable to initialize module pygltr");

    return;
}

/* ========================================================================== */

/*
 *                      E n d   o f   m o d u l e   P y g l t r
 */

/* ========================================================================== */
