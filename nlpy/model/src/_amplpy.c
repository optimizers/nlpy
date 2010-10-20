/* ====================================================
 * Generic Ampl interface to Python
 *
 * This is version v0.2
 *
 * Dominique Orban                  Michael Friedlander
 * All rights reserved              Chicago, March 2003
 * ====================================================
 */

#include "Python.h"                /* Main Python header file */

#include "numpy/arrayobject.h"     /* NumPy header */

#include "spmatrix.h"
#include "spmatrix_api.h"
#include "ll_mat.h"

#include <math.h>
#include "asl_pfgh.h"              /* Ampl library headers    */
#include "getstub.h"
#include "jacpdim.h"               /* For partially-separable structure */
/* Various DEFINEs */

#define ONE  (real)1.0

#define SYMMETRIC 1    /* Symmetric SpMatrix */
#define GENERAL   0    /* General   SpMatrix */

/* ========================================================================== */

/*
 *        P r o t o t y p e s   f o r   m o d u l e   f u n c t i o n s
 */

/* ========================================================================== */

void init_amplpy( void );
static PyObject *AmplPy_Init(          PyObject *self, PyObject *args );
static PyObject *AmplPy_Terminate(     PyObject *self, PyObject *args );
static PyObject *AmplPy_WriteSolution( PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Obj_Type(  PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Dimension( PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_nnzj(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_nnzh(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_ConType(   PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_obj(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Grad_obj(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_cons(     PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_ci(       PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_gi(       PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_sgi(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_sgrad(    PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_cost(     PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_row(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_J(        PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_A(        PyObject *self, PyObject *args );
static PyObject *AmplPy_Eval_H(        PyObject *self, PyObject *args );
static PyObject *AmplPy_Prod_Hv(       PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_x0(        PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_pi0(       PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Lvar(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Uvar(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Lcon(      PyObject *self, PyObject *args );
static PyObject *AmplPy_Get_Ucon(      PyObject *self, PyObject *args );
static PyObject *AmplPy_IsLP(          PyObject *self, PyObject *args );

static int Ampl_Init( void );

/* Ampl driver specific declarations */

#define CHR (char*)    /* To avoid some warning messages */
static ASL_pfgh *asl;  /* Main ASL structure */

static FILE *ampl_file  =  NULL;   /* Connection with Ampl nl file */
static int   ampl_file_open = 0;   /* Number of open files counter */
static int   ampl_written_sol = 0; /* Indicates whether solution was written */

/*
 * Keywords must appear in alphabetical order.
 * Normally, AmplPy is not called from the command line.
 */
static keyword keywds[] = {};

static Option_Info Oinfo = { CHR"amplpy", CHR"AmplPy", CHR"amplpy_options",
                             keywds, nkeywds, 0, CHR"0.2", 0, 0, 0, 0, 0,
                             20101019 };

/* ========================================================================== */

/*
 *                    M o d u l e   f u n c t i o n s
 */

/* ========================================================================== */

static char AmplPy_Init_Doc[] = "Read in problem.";

static PyObject *AmplPy_Init( PyObject *self, PyObject *args ) {

    char  **argv;
    char   *stub;    /* file name containing Ampl problem */

    if( ampl_file_open ) {
        PyErr_SetString(PyExc_ValueError, "A file is already open.");
        return NULL;
    }

    /* Arguments are passed by Python -- need to parse them first */
    /* Suppose for now that only 'stub' was passed ... */

    if( ! PyArg_ParseTuple( args, "s", &stub ) ) {
        PyErr_SetString(PyExc_ValueError, "Use: ampl_init(stub).");
        return NULL;
    }

    if( !(argv = (char **)calloc( 3, sizeof( char* ) )) )        return NULL;
    if( !(argv[0] = malloc( 7*sizeof( char ) )) )                return NULL;
    if( !(argv[1] = malloc( (strlen(stub)+1)*sizeof( char ) )) ) return NULL;
    argv[2] = NULL;
    strcpy( argv[0], "amplpy" );
    strcpy( argv[1], stub );

    /* Initialize main ASL structure */
    asl  = (ASL_pfgh*)ASL_alloc( ASL_read_pfgh );
    if( !asl ) return NULL;
    if( (stub = getstub( &argv, &Oinfo )) == NULL ) return NULL;
    ampl_file = jac0dim( stub, (fint)strlen( stub ) );

    /* Get command-line options */
    getopts( argv, &Oinfo );

    /* Allocate and initialize structures to hold problem data */
    if( Ampl_Init( ) ) return NULL;

    /* Read in ASL structure */
    pfgh_read( ampl_file , 0 );

    /* Specify that a file is now open */
    ampl_file_open = 1;

    /* Return to caller */
    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static int Ampl_Init( void ) {

    /* Allocate room to store problem data */
    if( ! (X0    = (real *)M1alloc(n_var*sizeof(real))) ) return -1;
    if( ! (LUv   = (real *)M1alloc(n_var*sizeof(real))) ) return -2;
    if( ! (Uvx   = (real *)M1alloc(n_var*sizeof(real))) ) return -3;
    if( ! (pi0   = (real *)M1alloc(n_con*sizeof(real))) ) return -4;
    if( ! (LUrhs = (real *)M1alloc(n_con*sizeof(real))) ) return -5;
    if( ! (Urhsx = (real *)M1alloc(n_con*sizeof(real))) ) return -6;

    /* Set Ampl reading options */
    want_xpi0 = 3;           /* Read primal and dual estimates */

    return 0;
}

/* ========================================================================== */

static char AmplPy_WriteSolution_Doc[] = "Output solution.";

static PyObject *AmplPy_WriteSolution( PyObject *self, PyObject *args ) {

    /* Output solution x and z passed as arguments */
    PyArrayObject *a_x, *a_z;
    char          *msg;

    /* We read the two arrays x and z, and a message */

    if( !PyArg_ParseTuple( args, "O!O!s",
               &PyArray_Type, &a_x,
               &PyArray_Type, &a_z, &msg ) )
        return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;
    if( a_z->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    if( !a_z ) return NULL;                       /* conversion error */
    if( a_z->nd != 1 ) return NULL;        /* z must have 1 dimension */
    if( a_z->dimensions[0] != n_con ) return NULL;  /* and size n_con */

    PyArray_XDECREF( a_z );

    Oinfo.wantsol = 9;   /* Suppress message echo, force .sol writing */

    /* Output solution */
    write_sol( CHR msg, (real *)a_x->data, (real *)a_z->data, &Oinfo );

    /* Indicate that a solution has been written */
    ampl_written_sol = 1;

    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static char AmplPy_Terminate_Doc[] = "Shut down.";

static PyObject *AmplPy_Terminate( PyObject *self, PyObject *args ) {

    /* Free Ampl data structures */
    //if( X0 ) free( X0  );
    //if( LUrhs ) free( LUrhs );
    //if( Uvx ) free( Uvx );
    //if( pi0 ) free( pi0 );
    //if( Urhsx ) free( Urhsx );
    //if( LUv ) free( LUv );
    ASL_free( (ASL **)(&asl) );
  //free( asl );

    /* Output a dummy solution if none has been output */
    //if( !ampl_written_sol )
    //    write_sol( CHR"Connection closed.", 0, 0, &Oinfo );

    /* Restore open file counter */
    ampl_file_open  = 0;

    /* Return to caller */
    Py_INCREF( Py_None );
    return Py_None;
}

/* ========================================================================== */

static char AmplPy_Get_Obj_Type_Doc[] = "Determine whether problem is a\
 minimization or maximization problem.";

static PyObject *AmplPy_Get_Obj_Type( PyObject *self, PyObject *args ) {

  /* objtype[0]=0 means that we have a minimization problem. */
  return Py_BuildValue("i", objtype[0]);
}

/* ========================================================================== */

static char AmplPy_Get_Dimension_Doc[] = "Obtain n and m.";

static PyObject *AmplPy_Get_Dimension( PyObject *self, PyObject *args ) {

    /* Ampl stores #variables and #constraint in n_var and n_con respectively */
    return Py_BuildValue( "ii", n_var, n_con );
}

/* ========================================================================== */

static char AmplPy_Get_nnzj_Doc[] = "Obtain nnzj.";

static PyObject *AmplPy_Get_nnzj( PyObject *self, PyObject *args ) {

    /* Ampl stores the #nonzeros in the Jacobian in nzc */
    return Py_BuildValue( "i", nzc );

}

/* ========================================================================== */

static char AmplPy_Get_nnzh_Doc[] = "Obtain nnzh.";

static PyObject *AmplPy_Get_nnzh( PyObject *self, PyObject *args ) {

    int nnzh;

    /* sphsetup( ) returns the #nonzeros in the Hessian of the Lagrangian */
    nnzh = (int)sphsetup( -1, 1, 1, 1 );
    return Py_BuildValue( "i", nnzh );

}

/* ========================================================================== */

static char AmplPy_Get_ConType_Doc[] = "Obtain indices of each type of constraints";

static PyObject *AmplPy_Get_ConType( PyObject *self, PyObject *args ) {

    PyObject *lin, *nln, *net, *item;
    int  i;

    /* Prepare lists to hold constraint indices */
    if( !(lin = PyList_New( 0 )) ) return NULL;
    if( !(nln = PyList_New( 0 )) ) return NULL;
    if( !(net = PyList_New( 0 )) ) return NULL;

    /* In Ampl, constraints are stored in the following order
     *     [1] nonlinear constraints,
     *     [2] network   constraints,
     *     [3] linear    constraints.
     */

    for( i = 0; i < n_con - nlc - nlnc; i++ ) {
        item = (PyObject *)PyInt_FromLong( nlc + nlnc + i );
        if( PyList_Append( lin, item ) ) {
            Py_DECREF( lin );
            return NULL;
        }
    }

    for( i = 0; i < nlc; i++ ) {
        item = (PyObject *)PyInt_FromLong( i );
        if( PyList_Append( nln, item ) ) {
            Py_DECREF( nln );
            return NULL;
        }
    }

    for( i = 0; i < nlnc; i++ ) {
        item = (PyObject *)PyInt_FromLong( nlc + i );
        if( PyList_Append( net, item ) ) {
            Py_DECREF( net );
            return NULL;
        }
    }

    return Py_BuildValue( "OOO", lin, nln, net );
}

/* ========================================================================== */

static char AmplPy_IsLP_Doc[] = "Determines whether problem is an LP";

static PyObject *AmplPy_IsLP( PyObject *self, PyObject *args ) {

    /* Return 1 if problem is an LP and 0 otherwise */
    int islp = 1;
    if( nlo || nlc || nlnc ) islp = 0;
    return Py_BuildValue( "i", islp );
}

/* ========================================================================== */

static char AmplPy_Get_x0_Doc[] = "Obtain initial guess.";

static PyObject *AmplPy_Get_x0( PyObject *self, PyObject *args ) {

    /* Fetch initial point, stored in X0 by Ampl */

    PyArrayObject *a_x0;  /* Initial point as a Numeric array */
    int i;
    npy_intp dx0[1] = { n_var };
    real *px0;

    a_x0 = (PyArrayObject *)PyArray_SimpleNew( 1, dx0, NPY_FLOAT64 );
    if( a_x0 == NULL ) return NULL;
    px0  = (real *)a_x0->data;
    for( i=0; i<n_var; i++ )
        px0[i] = X0[i];

    return PyArray_Return( a_x0 );
}

/* ========================================================================== */

static char AmplPy_Get_pi0_Doc[] = "Obtain initial multipliers.";

static PyObject *AmplPy_Get_pi0( PyObject *self, PyObject *args ) {

    /* Fetch initial multipliers, stored in pi0 by Ampl */

    PyArrayObject *a_pi0;  /* Initial multipliers as a Numeric array */
    int i;
    npy_intp dpi0[1] = { n_con };
    real *ppi0;

    a_pi0 = (PyArrayObject *)PyArray_SimpleNew( 1, dpi0, NPY_FLOAT64 );
    if( a_pi0 == NULL ) return NULL;
    ppi0  = (real *)a_pi0->data;
    for( i=0; i<n_con; i++ )
        ppi0[i] = pi0[i];

    return PyArray_Return( a_pi0 );
}

/* ========================================================================== */

static char AmplPy_Get_Lvar_Doc[] = "Obtain lower bounds on x.";

static PyObject *AmplPy_Get_Lvar( PyObject *self, PyObject *args ) {

    /* Fetch lower bounds on x, stored in LUv by Ampl */

    PyArrayObject *a_luv;  /* Lower bounds as a Numeric array */
    int i;
    npy_intp dluv[1] = { n_var };
    real *pluv;

    a_luv = (PyArrayObject *)PyArray_SimpleNew( 1, dluv, NPY_FLOAT64 );
    if( a_luv == NULL ) return NULL;
    pluv  = (real *)a_luv->data;
    for( i=0; i<n_var; i++ )
        pluv[i] = LUv[i];

    return PyArray_Return( a_luv );
}

/* ========================================================================== */

static char AmplPy_Get_Uvar_Doc[] = "Obtain upper bounds on x.";

static PyObject *AmplPy_Get_Uvar( PyObject *self, PyObject *args ) {

    /* Fetch upper bounds on x, stored in Uvx by Ampl */

    PyArrayObject *a_uvx;  /* Lower bounds as a Numeric array */
    int i;
    npy_intp duvx[1] = { n_var };
    real *puvx;

    a_uvx = (PyArrayObject *)PyArray_SimpleNew( 1, duvx, NPY_FLOAT64 );
    if( a_uvx == NULL ) return NULL;
    puvx  = (real *)a_uvx->data;
    for( i=0; i<n_var; i++ )
        puvx[i] = Uvx[i];

    return PyArray_Return( a_uvx );
}

/* ========================================================================== */

static char AmplPy_Get_Lcon_Doc[] = "Obtain constraints lower bounds.";

static PyObject *AmplPy_Get_Lcon( PyObject *self, PyObject *args ) {

    /* Fetch lower bounds on constraints, stored in LUrhs by Ampl */

    PyArrayObject *a_lurhs;  /* Lower bounds as a Numeric array */
    int i;
    npy_intp dlurhs[1] = { n_con };
    real *plurhs;

    a_lurhs = (PyArrayObject *)PyArray_SimpleNew( 1, dlurhs, NPY_FLOAT64 );
    if( a_lurhs == NULL ) return NULL;
    plurhs  = (real *)a_lurhs->data;
    for( i=0; i<n_con; i++ )
        plurhs[i] = LUrhs[i];

    return PyArray_Return( a_lurhs );
}

/* ========================================================================== */

static char AmplPy_Get_Ucon_Doc[] = "Obtain constraints upper bounds.";

static PyObject *AmplPy_Get_Ucon( PyObject *self, PyObject *args ) {

    /* Fetch upper bounds on x, stored in Urhsx by Ampl */

    PyArrayObject *a_urhsx;  /* Lower bounds as a Numeric array */
    int i;
    npy_intp durhsx[1] = { n_con };
    real *purhsx;

    a_urhsx = (PyArrayObject *)PyArray_SimpleNew( 1, durhsx, NPY_FLOAT64 );
    if( a_urhsx == NULL ) return NULL;
    purhsx  = (real *)a_urhsx->data;
    for( i=0; i<n_con; i++ )
        purhsx[i] = Urhsx[i];

    return PyArray_Return( a_urhsx );
}

/* ========================================================================== */

static char AmplPy_Eval_obj_Doc[] = "Evaluate objective.";

static PyObject *AmplPy_Eval_obj( PyObject *self, PyObject *args ) {

    /* Evaluate the objective function at the point x passed as argument.
     * The point x is given in the form of an array.
     * For now, only support single objective.
     */

    PyArrayObject *a_x;   /* Current point as a Numeric Array */
    fint nerror = (fint)0;
    real f;

    /* Read a single array */

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_x ) ) return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    f = objval( 0, (real *)a_x->data, NULL );
    if( nerror ) return NULL;

    return Py_BuildValue( "d", f );

}

/* ========================================================================== */

static char AmplPy_Grad_obj_Doc[] = "Evaluate objective gradient.";

static PyObject *AmplPy_Grad_obj( PyObject *self, PyObject *args ) {

  /* Evaluate the objective function gradient at the point x passed as argument.
   * The point x is given in the form of an array.
   * For now, only support single objective.
   */

    PyArrayObject *a_x;   /* Current point as a Numeric Array */
    PyArrayObject *a_g;   /* Gradient of f as a Numeric Array */
    fint nerror = (fint)0;
    npy_intp dg[1] = { n_var };

    /* Read a single array */

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_x ) ) return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                     /* conversion error */
    if( a_x->nd != 1 ) return NULL;       /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    /* Make room to store the gradient of the objective */
    a_g = (PyArrayObject *)PyArray_SimpleNew( 1, dg, NPY_FLOAT64 );
    if( a_g == NULL ) return NULL;
    objgrd( 0, (real *)a_x->data, (real *)a_g->data, &nerror );
    if( nerror ) return NULL;

    return PyArray_Return( a_g );
}

/* ========================================================================== */

static char AmplPy_Eval_cons_Doc[] = "Evaluate constraints.";

static PyObject *AmplPy_Eval_cons( PyObject *self, PyObject *args ) {

    /* Evaluate the constraint functions at the point x passed as argument.
     * The point x is given in the form of an array.
     */

    PyArrayObject *a_x;   /* Current point as a Numeric Array */
    PyArrayObject *a_c;   /* Constraint vector as a Numeric Array */
    fint nerror = (fint)0;
    npy_intp dc[1] = { n_con };

    /* Read a single array */

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_x ) ) return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                     /* conversion error */
    if( a_x->nd != 1 ) return NULL;       /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    /* Store the constraints vector in a PyArrayObject */
    a_c = (PyArrayObject *)PyArray_SimpleNew( 1, dc, NPY_FLOAT64 );
    if( a_c == NULL ) {
        printf( "Eval_cons: Memory allocation error\n" );   
        return NULL;
    }
    conval( (real *)a_x->data, (real *)a_c->data, &nerror );
    if( nerror ) {
        printf( "There was an error while evaluating the constraints\n" );    
        return NULL;
    }

    return PyArray_Return( a_c );
}

/* ========================================================================== */

static char AmplPy_Eval_J_Doc[] = "Evaluate sparse constraints Jacobian.";

static PyObject *AmplPy_Eval_J( PyObject *self, PyObject *args ) {

    /* Evaluate the constraint Jacobian at the point x passed as argument.
     * The point x is given in the form of an array.
     */

    /* ----------------------------- */
    PyArrayObject *a_x;                   /* Current point as a Numeric Array */
    int    coord;     /* Determine whether coordinate or LL format is desired */

    /* Variables corresponding to coordinate format */
    PyArrayObject *a_J;             /* Constraint Jacobian as a Numeric Array */
    PyArrayObject *a_irow, *a_icol;    /* Coordinate arrays as Numeric Arrays */
    npy_intp       dJ[1];                             /* Dimension descriptor */
    long          *pirow, *picol; /* Temporaries for tranfer of irow and icol */

    /* Variables corresponding to LL format */
    PyObject *spJac=NULL;                             /* The sparse Jacobian. */
    real     *J;               /* Constraint Jacobian as returned by jacval() */
    int       irow, jcol;    /* Row and col indices of nonzero Jacobian elems */
    int       dim[2] = {n_con,n_var}; /* Dimensions of sparse Jacobian: m, n. */

    /* Misc */
    cgrad *cg;                                         /* Jacobian in the DAG */
    int    nnzj;    /* To allocate Jacobian and account for unconstrained pbs */
    fint   nerror = (fint)0;                                  /* Error flag   */
    int    PassedJ = 1;         /* Indicates whether matrix was passed or not */
    int    i;                                                   /* Loop index */
    /* ----------------------------- */

    /* Read an array and an integer, and possibly a Jacobian matrix */

    if( !PyArg_ParseTuple( args, "O!i|O", &PyArray_Type, &a_x, &coord, &spJac) )
      return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                     /* conversion error */
    if( a_x->nd != 1 ) return NULL;       /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    /* See if sparse matrix was passed as argument */
    if( !spJac ) PassedJ = 0;

    /* Determine room necessary for Jacobian */
    nnzj = n_con ? nzc : 1;

    if( coord ) {  /* Return Jacobian in coordinate format */

        dJ[0] = nnzj;
        a_J = (PyArrayObject *)PyArray_SimpleNew( 1, dJ, NPY_FLOAT64 );
        if( a_J == NULL ) return NULL;
        jacval( (real *)a_x->data, (real *)a_J->data, &nerror );
        if( nerror ) return NULL;

        /* Obtain arrays icol and irow */
        a_irow = (PyArrayObject *)PyArray_SimpleNew( 1, dJ, NPY_INT );
        if( a_irow == NULL ) return NULL;
        a_icol = (PyArrayObject *)PyArray_SimpleNew( 1, dJ, NPY_INT );
        if( a_icol == NULL ) return NULL;
        pirow  = (long *)a_irow->data;
        picol  = (long *)a_icol->data;

        for( i=0; i<n_con; i++ )
            for( cg = Cgrad[i]; cg; cg = cg->next ) {
                pirow[ cg->goff ] = (long)i;
                picol[ cg->goff ] = (long)(cg->varno);
            }

        /* Return the triple ( J, irow, icol ) */
        return Py_BuildValue( "OOO",
                              PyArray_Return( a_J ),
                              PyArray_Return( a_irow ),
                              PyArray_Return( a_icol ) );

    } else {  /* Return Jacobian in LL format */

        if( !PassedJ )
            spJac = SpMatrix_NewLLMatObject( dim, GENERAL, nnzj );

        J = (real *)Malloc( nnzj * sizeof( real ) );
    
        /* Evaluate Jacobian and load the data array. */
        jacval( (real *)a_x->data, J, &nerror );
        if( nerror ) {
            if( J ) free( J );
            return NULL;
        }
    
        /* Create sparse Jacobian structure. */
        for( i=0; i<n_con; i++ ) {
          for( cg = Cgrad[i]; cg; cg = cg->next ) {
            irow = (long)i;
            jcol = (long)(cg->varno);
            SpMatrix_LLMatSetItem((LLMatObject *)spJac, irow, jcol, J[cg->goff]);
          }
        }
        if( J ) free( J );
        if( !PassedJ )
            return spJac;  /* Return sparse Jacobian. */
        else {
            Py_INCREF( Py_None );
            return Py_None;
        }
    }
}

/* ========================================================================== */

static char AmplPy_Eval_ci_Doc[] = "Evaluate i-th constraint.";

static PyObject *AmplPy_Eval_ci( PyObject *self, PyObject *args ) {

    /* Evaluate the i-th constraint value at the point x passed as argument.
     * The point x is given in the form of an array.
     */

    PyArrayObject *a_x;      /* Current point as a Numeric Array */
    int    i;                /* Constraint index */
    fint   nerror = (fint)0; /* Error flag   */
    real ci_of_x;            /* ci(x) */

    /* We read the constraint index and the vector x */

    if( !PyArg_ParseTuple( args, "iO!", &i, &PyArray_Type, &a_x ) )
    return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    /* Check whether index i makes sense */
    if( i < 0 || i >= n_con ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */
   
    PyArray_XDECREF( a_x );

    /* Evaluate i-th constraint at x */
    ci_of_x = conival( i, (real *)a_x->data, &nerror );
    if( nerror ) return NULL;

    return Py_BuildValue( "d", ci_of_x );
}

/* ========================================================================== */

static char AmplPy_Eval_gi_Doc[] = "Evaluate i-th constraint gradient.";

static PyObject *AmplPy_Eval_gi( PyObject *self, PyObject *args ) {

    /* Evaluate the i-th constraint gradient at the point x passed as argument.
     * The point x is given in the form of an array.
     *
     * Note that it is possible to return a sparse gradient here.
     */

    PyArrayObject *a_x;       /* Current point as a Numeric Array */
    int    i;                 /* Loop index */
    fint   nerror = (fint)0;  /* Error flag   */
    PyArrayObject *a_gi;      /* grad ci(x) as a Numeric Array */
    npy_intp   dgi[1] = { n_var }; /* Dimension descriptor */

    /* We read the constraint index and the vector x */

    if( !PyArg_ParseTuple( args, "iO!", &i, &PyArray_Type, &a_x ) )
    return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    /* Check whether index i makes sense */
    if( i < 0 || i >= n_con ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */
   
    PyArray_XDECREF( a_x );

    /* Make room and evaluate i-th constraint at x */
    a_gi = (PyArrayObject *)PyArray_SimpleNew( 1, dgi, NPY_FLOAT64 );
    if( a_gi == NULL ) return NULL;
    congrd( i, (real *)a_x->data, (real *)a_gi->data, &nerror );
    if( nerror ) return NULL;

    /* Return dense gradient */
    return PyArray_Return( a_gi );
}

/* ========================================================================== */

static char AmplPy_Eval_cost_Doc[] = "Evaluate sparse cost vector.";

static PyObject *AmplPy_Eval_cost( PyObject *self, PyObject *args ) {
    
    /* Evaluate cost vector as a sparse vector.
     * To be used when problem is a linear program; this will be
     * faster than AmplPy_Eval_sgrad().
     */

    ograd    *og;
    PyObject *cost;
    PyObject *key, *val;

    cost = PyDict_New();
    if( !cost ) return NULL;

    for( og = Ograd[0]; og; og = og->next ) {
        key = (PyObject *)PyInt_FromLong( og->varno );
        val = (PyObject *)PyFloat_FromDouble( og->coef );
        PyDict_SetItem( cost, key, val );
    }
    
    return cost;
}

/* ========================================================================== */

static char AmplPy_Eval_sgrad_Doc[] = "Evaluate sparse objective gradient.";

static PyObject *AmplPy_Eval_sgrad( PyObject *self, PyObject *args ) {

    /* Evaluate sparse objective gradient at the point x passed as argument.
     * The point x is given in the form of an array.
     * The sparse gradient is returned as a dictionary.
     */

    PyArrayObject *a_x;        /* Current point as a Numeric Array */
    int     j;                 /* Loop index */
    fint    nerror = (fint)0;  /* Error flag   */
    double *grad_f;            /* Sparse grad f(x) */
    long    nzg;               /* Number of nonzeros in sparse gradient */
    ograd  *og;                /* Description of sparse gradient */

    PyObject *sg;
    PyObject *key, *val;

    /* We read the constraint index and the vector x */

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_x ) )
    return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */
   
    PyArray_XDECREF( a_x );

    /* Count number of nonzeros */
    nzg = 0;
    for( og = Ograd[0]; og; og = og->next ) nzg++;

    /* Make room and evaluate i-th constraint at x */
    grad_f = (double *)malloc( nzg * sizeof( double ) );
    if( !grad_f ) return NULL;
    objgrd( 0, (real *)a_x->data, grad_f, &nerror );
    if( nerror ) return NULL;

    /* Generate index array */
    sg = PyDict_New();
    if( !sg ) return NULL;
    j = 0;
    for( og = Ograd[0]; og; og = og->next ) {
        key = (PyObject *)PyInt_FromLong( og->varno );
        val = (PyObject *)PyFloat_FromDouble( grad_f[j] );
        PyDict_SetItem( sg, key, val );
        j++;
    }

    /* Return sparse gradient */
    return sg;
}

/* ========================================================================== */

static char AmplPy_Eval_row_Doc[] = "Evaluate i-th row of Jacobian for LPs.";

static PyObject *AmplPy_Eval_row( PyObject *self, PyObject *args ) {
    
    /* Evaluate the i-th constraint gradient as a sparse vector.
     * To be used when problem is a linear programming problem.
     * This will be faster than Eval_sgi().
     */

    cgrad    *cg;
    PyObject *row;
    PyObject *key, *val;
    int       i;

    /* Read constraint index */
    if( !PyArg_ParseTuple( args, "i", &i ) )
    return NULL;

    /* Check whether index i makes sense */
    if( i < 0 || i >= n_con ) return NULL;

    row = PyDict_New();
    if( !row ) return NULL;

    for( cg = Cgrad[i]; cg; cg = cg->next ) {
        key = (PyObject *)PyInt_FromLong( cg->varno );
        val = (PyObject *)PyFloat_FromDouble( cg->coef );
        PyDict_SetItem( row, key, val );
    }
    
    return row;
}

/* ========================================================================== */

static char AmplPy_Eval_A_Doc[] = "Evaluate Jacobian for LPs.";

static PyObject *AmplPy_Eval_A( PyObject *self, PyObject *args ) {

    long      irow, jcol;
    int       PassedJ = 1, nnzj, i;
    cgrad    *cg;
    PyObject *spJac = NULL;
    int       dim[2] = {n_con, n_var};

    if( !PyArg_ParseTuple( args, "|O", &spJac ) )
        return NULL;

    /* See if sparse matrix was passed as argument */
    if( !spJac ) PassedJ = 0;

    /* Determine room necessary for Jacobian */
    nnzj = n_con ? nzc : 1;

    if( !PassedJ )
        spJac = SpMatrix_NewLLMatObject( dim, GENERAL, nnzj );

    /* Create sparse Jacobian structure. */
    for( i=0; i<n_con; i++ ) {
        irow = (long)i;
        for( cg = Cgrad[i]; cg; cg = cg->next ) {
            jcol = (long)(cg->varno);
            SpMatrix_LLMatSetItem( (LLMatObject *)spJac, irow, jcol, cg->coef );
        }
    }
    if( !PassedJ )
        return spJac;  /* Return sparse Jacobian. */
    else {
        Py_INCREF( Py_None );
        return Py_None;
    }
}

/* ========================================================================== */

static char AmplPy_Eval_sgi_Doc[] = "Evaluate i-th sparse constraint gradient.";

static PyObject *AmplPy_Eval_sgi( PyObject *self, PyObject *args ) {

    /* Evaluate the i-th constraint sparse gradient at the point x passed as
     * argument.
     * The point x is given in the form of an array.
     * The sparse gradient is returned as a dictionary.
     */

    PyArrayObject *a_x;       /* Current point as a Numeric Array */
    int    i, j;              /* Loop index */
    fint   nerror = (fint)0;  /* Error flag   */
    double *grad_ci;          /* Sparse grad ci(x) */
    long  nzgi;               /* Number of nonzeros in sparse gradient */
    cgrad *cg;                /* Description of sparse gradient */

    PyObject *sgi;
    PyObject *key, *val;
    int       congrd_mode_save;

    /* We read the constraint index and the vector x */

    if( !PyArg_ParseTuple( args, "iO!", &i, &PyArray_Type, &a_x ) )
    return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    /* Check whether index i makes sense */
    if( i < 0 || i >= n_con ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */
   
    PyArray_XDECREF( a_x );

    /* Set sparse format for gradient */
    congrd_mode_save = asl->i.congrd_mode;
    asl->i.congrd_mode = 1;

    /* Count number of nonzeros in gi */
    nzgi = 0;
    for( cg = Cgrad[i]; cg; cg = cg->next ) nzgi++;

    /* Make room and evaluate i-th constraint at x */
    grad_ci = (double *)malloc( nzgi * sizeof( double ) );
    if( !grad_ci ) return NULL;
    congrd( i, (real *)a_x->data, grad_ci, &nerror );
    if( nerror ) return NULL;

    /* Generate index array */
    sgi = PyDict_New();
    if( !sgi ) return NULL;
    j = 0;
    for( cg = Cgrad[i]; cg; cg = cg->next ) {
        key = (PyObject *)PyInt_FromLong( cg->varno );
        val = (PyObject *)PyFloat_FromDouble( grad_ci[j] );
        PyDict_SetItem( sgi, key, val );
        j++;
    }

    // Restore gradient mode
    asl->i.congrd_mode = congrd_mode_save; 

    /* Return sparse gradient */
    return sgi;
}

/* ========================================================================== */

static char AmplPy_Eval_H_Doc[] = "Evaluate sparse upper triangle of Lagrangian Hessian.";

static PyObject *AmplPy_Eval_H( PyObject *self, PyObject *args ) {

    /* Evaluate the Hessian of the Lagrangian at the point (x,lambda) passed as
     * argument.
     * The points x and lambda are given in the form of arrays.
     *
     *      NOTE: .... why are we passing x ???
     *
     * In the future, we will want to be careful here, in case x has changed
     * but f(x), c(x) or J(x) have not yet been recomputed. In such a case, Ampl
     * has NOT updated the data structure for the Hessian, and it will still
     * hold the Hessian at the last point at which, f, c or J were evaluated !
     */

    /* ----------------------------- */
    PyArrayObject *a_x, *a_lambda;       /* Current points as a Numeric Array */
    int            coord;  /* Determine whether coord or LL format is desired */

    /* Variables corresponding to coordinate format */
    PyArrayObject *a_H;         /* Hessian of the Lagrangian as a Numpy Array */
    PyArrayObject *a_irow, *a_icol;      /* Coordinate arrays as Numpy Arrays */
    long          *pirow, *picol; /* Temporaries for tranfer of irow and icol */
    npy_intp       dH[1];                             /* Dimension descriptor */

    /* Variables corresponding to LL format */
    PyObject *spHess = NULL;                 /* The sparse, symmetric Hessian */
    real     *H;          /* Hessian of the Lagrangian as returned by sphes() */
    int       jrow, jcol;     /* Row and col indices of nonzero Hessian elems */
    int       dim[2] = {n_var, n_var};     /* Dimensions of the sparse Hessian*/

    /* Misc */
    real   OW[1];         /* Objective type: support single objective for now */
    int    nnzh;     /* Number of nonzeros in sparse upper triangular Hessian */
    int    PassedH = 1;         /* Indicates whether matrix was passed or not */
    int    i, j, k;                                           /* Loop indices */
    /* ----------------------------- */

    /* Read two arrays and an integer */

    if( !PyArg_ParseTuple( args, "O!O!i|O",
               &PyArray_Type, &a_x,
               &PyArray_Type, &a_lambda, &coord, &spHess ) )
    return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;
    if( a_lambda->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_lambda ) return NULL;                       /* conversion error */
    if( a_lambda->nd != 1 ) return NULL;   /* lambda must have 1 dimension */
    if( a_lambda->dimensions[0] != n_con ) return NULL;  /* and size n_con */

    PyArray_XDECREF( a_lambda );

    /* See if matrix was passed as argument */
    if( !spHess ) PassedH = 0;

    /* Determine room for Hessian and multiplier sign. */
    nnzh   = (int)sphsetup( -1, 1, 1, 1 );
    OW[0]  = objtype[0] ? -ONE : ONE;   /* Indicates min/max-imization */

    if( coord ) { /* Return Hessian in coordinate format */

        dH[0]  = nnzh;
        a_H = (PyArrayObject *)PyArray_SimpleNew( 1, dH, NPY_FLOAT64 );
        if( a_H == NULL ) return NULL;
        /* Evaluate Hessian */
        sphes( (real *)a_H->data, -1, OW, (real *)a_lambda->data );

        /* Obtain row and column indices */
        a_irow = (PyArrayObject *)PyArray_SimpleNew( 1, dH, NPY_INT );
        if( a_irow == NULL ) return NULL;
        a_icol = (PyArrayObject *)PyArray_SimpleNew( 1, dH, NPY_INT );
        if( a_icol == NULL ) return NULL;
        pirow  = (long *)a_irow->data;
        picol  = (long *)a_icol->data;

        k = 0;
        for( i=0; i<n_var; i++ ) {
            for( j=sputinfo->hcolstarts[i]; j<sputinfo->hcolstarts[i+1]; j++ ) {
                pirow[k] = sputinfo->hrownos[j];
                picol[k] = i;
                k++;
            }
        }

        /* Return the triple ( H, irow, icol ) */
        return Py_BuildValue( "OOO",
                              PyArray_Return( a_H ),
                              PyArray_Return( a_irow ),
                              PyArray_Return( a_icol ) );

    } else { /* Return Hessian in LL format */

        H = (real *)Malloc( nnzh * sizeof( real ) );
        if( !H ) return NULL;
        sphes( H, -1, OW, (real *)a_lambda->data );

        /* Allocate sparse symmetric Hessian data structure */
        if( !PassedH ) {
          spHess = SpMatrix_NewLLMatObject( dim, SYMMETRIC, nnzh );
          if( !spHess ) return NULL;
        }

        /* Transfer H into the PySparse LL-matrix.
         * Ampl returns the upper triangle of H. PySparse wants the
         * lower triangle. Accommodate by reversing indices.
         */
        for( i=0; i<n_var; i++ )
          for( j=sputinfo->hcolstarts[i]; j<sputinfo->hcolstarts[i+1]; j++ )
            SpMatrix_LLMatSetItem((LLMatObject *)spHess,
                                  i,
                                  sputinfo->hrownos[j],
                                  H[j]);

        if( H ) free( H );
        if( !PassedH )
            return spHess;  /* Return sparse Hessian. */
        else {
            Py_INCREF( Py_None );
            return Py_None;
        }
    }
}

/* ========================================================================== */

static char AmplPy_Prod_Hv_Doc[] = "Compute matrix-vector product Hv of Lagrangian Hessian times a vector.";

static PyObject *AmplPy_Prod_Hv( PyObject *self, PyObject *args ) {

    PyArrayObject *a_v, *a_lambda, *a_Hv;
    real           OW[1];
    int            nnzh;
    npy_intp       dHv[1];

    /* We read the vector v and the multipliers lambda */
    if( !PyArg_ParseTuple( args, "O!O!",
               &PyArray_Type, &a_lambda, &PyArray_Type, &a_v ) )
    return NULL;
    if( a_v->descr->type_num != NPY_FLOAT64 ) return NULL;
    if( a_lambda->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_v ) return NULL;                            /* conversion error */
    if( a_v->nd != 1 ) return NULL;             /* v must have 1 dimension */
    if( a_v->dimensions[0] != n_var ) return NULL;       /* and size n_var */

    if( !a_lambda ) return NULL;                       /* conversion error */
    if( a_lambda->nd != 1 ) return NULL;   /* lambda must have 1 dimension */
    if( a_lambda->dimensions[0] != n_con ) return NULL;
                                                         /* and size n_con */
    PyArray_XDECREF( a_v );
    PyArray_XDECREF( a_lambda );

    /* Determine room for Hessian and multiplier sign. */
    nnzh   = (int)sphsetup( -1, 1, 1, 1 );
    OW[0]  = objtype[0] ? -ONE : ONE;   /* Indicates min/max-imization */
    dHv[0]  = n_var;
    a_Hv = (PyArrayObject *)PyArray_SimpleNew( 1, dHv, NPY_FLOAT64 );
    if( a_Hv == NULL ) return NULL;

    /* Evaluate matrix-vector product Hv */
    hvcomp((real *)a_Hv->data, (real *)a_v->data, 0, OW, (real *)a_lambda->data);
    /* Return Hv */
    return Py_BuildValue( "O", PyArray_Return( a_Hv ) );
}

/* ========================================================================== */

static char AmplPy_Set_x_Doc[] = "Declare x as current primal value.";

static PyObject *AmplPy_Set_x( PyObject *self, PyObject *args ) {

    /* Call xknown() with given x as argument, to prevent subsequent calls to
     * objval, objgrad, etc., to check whether their argument has changed since
     * the last call. Users must not forget to call Unset_x when they are
     * finished, and before changing the value of x, or to call Set_x again with
     * an updated value of x.
     */

    PyArrayObject *a_x;   /* Current point as a Numeric Array */

    /* Read a single array */

    if( !PyArg_ParseTuple( args, "O!", &PyArray_Type, &a_x ) ) return NULL;
    if( a_x->descr->type_num != NPY_FLOAT64 ) return NULL;

    if( !a_x ) return NULL;                       /* conversion error */
    if( a_x->nd != 1 ) return NULL;        /* x must have 1 dimension */
    if( a_x->dimensions[0] != n_var ) return NULL;  /* and size n_var */

    PyArray_XDECREF( a_x );

    xknown((double *)a_x->data);

    Py_INCREF(Py_None);
    return Py_None;

}

/* ========================================================================== */

static char AmplPy_Unset_x_Doc[] = "Release current primal value.";

static PyObject *AmplPy_Unset_x( PyObject *self, PyObject *args ) {

    /* Call xunknown() to release current primal value and force subsequent calls
     * to objval, objgrad, etc., to check whether their argument has changed since
     * the last call.
     */

  xunknown();

  Py_INCREF(Py_None);
  return Py_None;

}

/* ========================================================================== */

/*
 *           D e f i n i t i o n   o f   A m p l P y   m e t h o d s
 */

/* ========================================================================== */

static PyMethodDef AmplPyMethods[] = {
  {"ampl_init", AmplPy_Init,          METH_VARARGS, AmplPy_Init_Doc          },
  {"ampl_shut", AmplPy_Terminate,     METH_VARARGS, AmplPy_Terminate_Doc     },
  {"ampl_sol",  AmplPy_WriteSolution, METH_VARARGS, AmplPy_WriteSolution_Doc },
  {"obj_type",  AmplPy_Get_Obj_Type,  METH_VARARGS, AmplPy_Get_Obj_Type_Doc  },
  {"get_dim",   AmplPy_Get_Dimension, METH_VARARGS, AmplPy_Get_Dimension_Doc },
  {"get_nnzj",  AmplPy_Get_nnzj,      METH_VARARGS, AmplPy_Get_nnzj_Doc      },
  {"get_nnzh",  AmplPy_Get_nnzh,      METH_VARARGS, AmplPy_Get_nnzh_Doc      },
  {"get_CType", AmplPy_Get_ConType,   METH_VARARGS, AmplPy_Get_ConType_Doc   },
  {"get_x0",    AmplPy_Get_x0,        METH_VARARGS, AmplPy_Get_x0_Doc        },
  {"get_pi0",   AmplPy_Get_pi0,       METH_VARARGS, AmplPy_Get_pi0_Doc       },
  {"get_Lvar",  AmplPy_Get_Lvar,      METH_VARARGS, AmplPy_Get_Lvar_Doc      },
  {"get_Uvar",  AmplPy_Get_Uvar,      METH_VARARGS, AmplPy_Get_Uvar_Doc      },
  {"get_Lcon",  AmplPy_Get_Lcon,      METH_VARARGS, AmplPy_Get_Lcon_Doc      },
  {"get_Ucon",  AmplPy_Get_Ucon,      METH_VARARGS, AmplPy_Get_Ucon_Doc      },
  {"eval_obj",  AmplPy_Eval_obj,      METH_VARARGS, AmplPy_Eval_obj_Doc      },
  {"grad_obj",  AmplPy_Grad_obj,      METH_VARARGS, AmplPy_Grad_obj_Doc      },
  {"eval_cons", AmplPy_Eval_cons,     METH_VARARGS, AmplPy_Eval_cons_Doc     },
  {"eval_ci",   AmplPy_Eval_ci,       METH_VARARGS, AmplPy_Eval_ci_Doc       },
  {"eval_gi",   AmplPy_Eval_gi,       METH_VARARGS, AmplPy_Eval_gi_Doc       },
  {"eval_sgi",  AmplPy_Eval_sgi,      METH_VARARGS, AmplPy_Eval_sgi_Doc      },
  {"eval_sgrad", AmplPy_Eval_sgrad,    METH_VARARGS, AmplPy_Eval_sgrad_Doc   },
  {"eval_cost", AmplPy_Eval_cost,     METH_VARARGS, AmplPy_Eval_cost_Doc     },
  {"eval_row",  AmplPy_Eval_row,      METH_VARARGS, AmplPy_Eval_row_Doc      },
  {"eval_J",    AmplPy_Eval_J,        METH_VARARGS, AmplPy_Eval_J_Doc        },
  {"eval_A",    AmplPy_Eval_A,        METH_VARARGS, AmplPy_Eval_A_Doc        },
  {"eval_H",    AmplPy_Eval_H,        METH_VARARGS, AmplPy_Eval_H_Doc        },
  {"H_prod",    AmplPy_Prod_Hv,       METH_VARARGS, AmplPy_Prod_Hv_Doc       },
  {"is_lp",     AmplPy_IsLP,          METH_VARARGS, AmplPy_IsLP_Doc          },
  {"set_x",     AmplPy_Set_x,         METH_VARARGS, AmplPy_Set_x_Doc         },
  {"unset_x",   AmplPy_Unset_x,       METH_VARARGS, AmplPy_Unset_x_Doc       },
  {NULL,        NULL,                 0,            NULL                     }
};

/* ========================================================================== */

void init_amplpy( void ) {

    (void)Py_InitModule3( "_amplpy", AmplPyMethods, "Python/Ampl interface" );

    import_array( );         /* Initialize the Numarray module. */
    import_spmatrix( );      /* Initialize the PySparse module. */

    /* Check for errors */
    if (PyErr_Occurred())
    Py_FatalError("Unable to initialize module amplpy");

    return;
}

/* ========================================================================== */

/*
 *                 E n d   o f   m o d u l e   A m p l P y
 */

/* ========================================================================== */
