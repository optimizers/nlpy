
/* ========================================================================== */

/*
 *                              P y c s r c h
 *  
 *  P y t h o n   i n t e r f a c e   t o   t h e   C S R C H   p a c k a g e
 */

/* ========================================================================== */

#include "Python.h"                /* Main Python header file */
#include "numpy/arrayobject.h"     /* NumPy header */
#include "string.h"

/* ========================================================================== */

/*
 *   D e f i n i t i o n   o f   P y c s r c h   c o n t e x t   o b j e c t
 */

/* ========================================================================== */

staticforward PyTypeObject PycsrchType;   /* Precise definition appears below */

typedef struct PycsrchObject {
    PyObject_VAR_HEAD
    double     fk;                                 /* Initial objective value */
    double     gkdk;                                         /* Initial slope */
    double     stp;                                     /* Current steplength */
    double     ftol, gtol, xtol;                    /* Convergence tolerances */
    double     stpmin, stpmax;                             /* Current bracket */
    int        isave[3];                                /* Integer work array */
    double     dsave[13];                                /* Double work array */
    char       task[61];                          /* Current task description */
} PycsrchObject;

/* Support different Fortran compilers */
#ifdef _AIX
#define FUNDERSCORE(a) a
#else
#define FUNDERSCORE(a) a##_
#endif

#define DCSRCH     FUNDERSCORE(dcsrch)
#define MAX(a,b)   (a) > (b) ? (a) : (b)
#define PycsrchObject_Check(v)  ((v)->ob_type == &PycsrchType)

/* ========================================================================== */

DL_EXPORT( void ) init_pycsrch( void );
static PycsrchObject *NewPycsrchObject( double ftol, double gtol, double xtol,
                                        double stp, double stpmin,
                                        double stpmax );
static PyObject *Pycsrch_Init( PyObject *self, PyObject *args );
static PyObject *Pycsrch_csrch( PycsrchObject *self, PyObject *args );
static void Pycsrch_dealloc( PycsrchObject *self );
static PyObject *Pycsrch_getattr( PycsrchObject *self, char *name );

/* the last artifical argument l to DCSRCH is the length of
 * the string 'task'. When passing a string from C to Fortran, the
 * length of the string should be appened, as a long integer.
 */
extern void DCSRCH( double *stp, double *f, double *g, double *ftol,
                    double *gtol, double *xtol, char *task,
                    double *stpmin, double *stpmax, int *isave,
                    double *dsave, long l );

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */

static PycsrchObject *NewPycsrchObject( double ftol,   double gtol,
                                        double xtol,   double stp,
                                        double stpmin, double stpmax ) {

    PycsrchObject *self;

    /* Create new instance of object */
    if( !(self = PyObject_New( PycsrchObject, &PycsrchType ) ) )
        return NULL; //PyErr_NoMemory( );

    /* Populate Pycsrch data structure */
    self->ftol = ftol;
    self->gtol = gtol;
    self->xtol = xtol;
    self->stp  = stp;
    strcpy( self->task, "START" );
    self->stpmin = stpmin;
    self->stpmax = stpmax;
    
    return self;
}

/* ========================================================================== */

static char Pycsrch_csrch_Doc[] = "Perform More-Thuente linesearch";

static PyObject *Pycsrch_csrch( PycsrchObject *self, PyObject *args ) {

    double f, slope;
    long   l;
    int    i;

    /* Input must be f(xk + step dk) and the slope of f at xk + step dk
     * in the direction dk
     */
    if( !PyArg_ParseTuple( args, "dd", &f, &slope ) ) return NULL;
    self->fk = f;
    self->gkdk = slope;

    /* Convert string so it is usable and modifiable by Fortran */
    l = strlen( self->task );
    for( i = l; i < 61; i++ ) self->task[i] = ' ';

    /* Perform linesearch */
    DCSRCH( &(self->stp),
            &(self->fk),
            &(self->gkdk),
            &(self->ftol),
            &(self->gtol),
            &(self->xtol),
            self->task,
            &(self->stpmin),
            &(self->stpmax),
            self->isave,
            self->dsave,
            60L );        /* length of string self->task */

    /* Convert string back so it is usable in C */
    for( i = 60; self->task[i] == ' '; i-- );
    self->task[i+1] = '\0';

    return Py_BuildValue( "ds", self->stp, self->task );
}

/* ========================================================================== */

/* This is necessary as Pycsrch_csrch takes a PycsrchObject* as argument */

static PyMethodDef Pycsrch_special_methods[] = {
    { "csrch", (PyCFunction)Pycsrch_csrch, METH_VARARGS, Pycsrch_csrch_Doc },
    { NULL,    NULL,                       0,            NULL              }
};

/* ========================================================================== */

static char Pycsrch_Init_Doc[] = "Initialize More-Thuente linesearch";

static PyObject *Pycsrch_Init( PyObject *self, PyObject *args ) {

    /* Input must be ftol, gtol, xtol, stp, stpmin, stpmax */

    PycsrchObject  *rv;                    /* Return value */
    double          ftol, gtol, xtol, stp, stpmin, stpmax;

    if( !PyArg_ParseTuple( args, "dddddd",
                           &ftol, &gtol, &xtol, &stp, &stpmin, &stpmax ) )
        return NULL;

    /* Spawn new Pycsrch Object */
    rv = NewPycsrchObject( ftol, gtol, xtol, stp, stpmin, stpmax );
    if( rv == NULL ) return NULL;

    return (PyObject *)rv;
}

/* ========================================================================== */

/*
 *    D e f i n i t i o n   o f   P y c s r c h   c o n t e x t   t y p e
 */

/* ========================================================================== */

static PyTypeObject PycsrchType = {
    PyObject_HEAD_INIT(NULL)
    0,
    "pycsrch_context",
    sizeof(PycsrchObject),
    0,
    (destructor)Pycsrch_dealloc,  /* tp_dealloc */
    0,                            /* tp_print */
    (getattrfunc)Pycsrch_getattr, /* tp_getattr */
    0,                            /* tp_setattr */
    0,                            /* tp_compare */
    0,                            /* tp_repr */
    0,                            /* tp_as_number*/
    0,                            /* tp_as_sequence*/
    0,                            /* tp_as_mapping*/
    0,                            /* tp_hash */
    0,                            /* tp_call*/
    0,                            /* tp_str*/
    0,                            /* tp_getattro*/
    0,                            /* tp_setattro*/
    0,                            /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,           /* tp_flags*/
    "PyCsrch Context Object",     /* tp_doc */
};

/* ========================================================================== */

/*
 *        D e f i n i t i o n   o f   P y c s r c h   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef PycsrchMethods[] = {
    { "Init",   Pycsrch_Init,    METH_VARARGS, Pycsrch_Init_Doc   },
    { NULL,     NULL,            0,            NULL               }
};

/* ========================================================================== */

static void Pycsrch_dealloc( PycsrchObject *self ) {

    PyObject_Del(self);
}

/* ========================================================================== */

static PyObject *Pycsrch_getattr( PycsrchObject *self, char *name ) {

    if( strcmp( name, "shape" ) == 0 )
        return Py_BuildValue( "(i,i)", 0, 0 );
    if( strcmp( name, "nnz" ) == 0 )
        return Py_BuildValue( "i", 0 );
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
    return Py_FindMethod( Pycsrch_special_methods, (PyObject *)self, name );
}

/* ========================================================================== */

DL_EXPORT( void ) init_pycsrch( void ) {

    PyObject *m, *d;

    if( PyType_Ready( &PycsrchType ) < 0 ) return;

    m = Py_InitModule3( "_pycsrch",
                        PycsrchMethods,
                        "Python interface to CSRCH" );

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "PycsrchType", (PyObject *)&PycsrchType);

    import_array( );         /* Initialize the Numarray module. */

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("Unable to initialize module pycsrch");

    return;
}

/* ========================================================================== */

