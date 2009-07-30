/* $Id: ma27_lib.c 84 2008-09-15 02:49:23Z d-orban $ */

#include <cblas.h>
#include "ma27.h"

#ifdef __cplusplus
extern "C" {   /* To prevent C++ compilers from mangling symbols */
#endif

/* -------------------------------------------------- */

static double
cblas_dnrm_infty( const int N, const double *X, const int incX ) {
    return fabs( X[ cblas_idamax( N, X, incX ) ] );
}

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "MA27_Initialize"
    Ma27_Data *Ma27_Initialize( int nz, int n, FILE *logfile ) {

        /* Call initialize subroutine MA27ID and set defaults */
        Ma27_Data *ma27 = (Ma27_Data *)NLPy_Calloc( 1, sizeof(Ma27_Data) );

        LOGMSG( " MA27 :: Initializing..." );
        ma27->logfile   = logfile;
        ma27->n         = n;
        ma27->nz        = nz;
        ma27->fetched   = 0;
        ma27->la        = ceil( 1.2 * nz );
        ma27->liw       = imax( ceil( 1.2 * ( 2*nz + 3*n + 1 )), LIW_MIN );
        ma27->irn       = (int *)NLPy_Calloc( nz, sizeof(int) );
        ma27->icn       = (int *)NLPy_Calloc( nz, sizeof(int) );
        ma27->iw        = (int *)NLPy_Calloc( ma27->liw, sizeof(int) );
        ma27->ikeep     = (int *)NLPy_Calloc( 3 * n, sizeof(int) );
        ma27->iw1       = (int *)NLPy_Calloc( 2 * n, sizeof(int) );
        ma27->factors   = (double *)NLPy_Calloc( ma27->la, sizeof(double) );
        // Don't allocate the residual for NLPy --- will come from Python
        //ma27->residual  = (double *)NLPy_Calloc( n, sizeof(double) );

        MA27ID( ma27->icntl, ma27->cntl );
        ma27->icntl[0] = 0;  // Stream for error messages.
        ma27->icntl[1] = 0;  // Stream for diagnotic messages.
        ma27->icntl[2] = 0;  // Verbosity: 0=none, 1=partial, 2=full

        LOGMSG( " done\n" );
        return ma27;
    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ma27_Analyze"
    int Ma27_Analyze( Ma27_Data *ma27, int iflag ) {

        int n = ma27->n, finished = 0, error;
        LOGMSG( " MA27 :: Analyzing\n" );

        ma27->iflag = iflag;

        /* Unpack data structure and call MA27AD */
        while( !finished ) {

            MA27AD(&(ma27->n), &(ma27->nz), ma27->irn, ma27->icn,
                   ma27->iw, &(ma27->liw), ma27->ikeep, ma27->iw1,
                   &(ma27->nsteps), &(ma27->iflag), ma27->icntl,
                   ma27->cntl, ma27->info, &(ma27->ops) );

            error = ma27->info[0];

            if( !error )
                finished = 1;
            else {
                error = Process_Error_Code( ma27, error );
                if( error != -3 && error != -4 ) return error;
            }
        }

        /* Adjust size of factors (if necessary) */
        if( ma27->info[4] > ma27->nz ) {
            ma27->la = ceil( 1.2 * ma27->info[4] );
            NLPy_Free( ma27->factors );
            ma27->factors = (double *)NLPy_Calloc( ma27->la, sizeof(double) );
        }

        /* Adjust size of w1 (if necessary) */
        if( ma27->nsteps > 2 * ma27->n ) {
            NLPy_Free( ma27->iw1 );
            ma27->iw1 = (int *)NLPy_Calloc( ma27->nsteps, sizeof(int) );
        }

        /* For now we assume the front size is maximal. */
        ma27->w = (double *)NLPy_Calloc( n, sizeof(double) );
        if( ! ma27->w ) return -10;

        return 0;
    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ma27_Factorize"
    int Ma27_Factorize( Ma27_Data *ma27, double A[] ) {

        int finished = 0, error;
        double pTol, new_pTol = PIV_MIN;
        LOGMSG( " MA27 :: Factorizing..." );
        
        /* Copy A into factors. */
        cblas_dcopy( ma27->nz, A, 1, ma27->factors, 1 );

        /* Unpack data structure and call MA27BD */
        while( !finished ) {
            LOGMSG( " calling ma27bd... " );

            MA27BD( &(ma27->n), &(ma27->nz), ma27->irn, ma27->icn,
                    ma27->factors, &(ma27->la), ma27->iw, &(ma27->liw),
                    ma27->ikeep, &(ma27->nsteps), &(ma27->maxfrt),
                    ma27->iw1, ma27->icntl, ma27->cntl, ma27->info );

            error = ma27->info[0];
            pTol = ma27->cntl[0];

            if( !error ) {
                LOGMSG( "completed" );
                finished = 1;
            } else if(   (error == 3 || error == -5)    // Singular
                      && (pTol <= PIV_MAX) ) {          // pTol less than max
                /* Adjust pivot tolerance */
                if( pTol == 0.0 )
                    pTol =  1.0e-6;
                else {
                    new_pTol = fmin( PIV_MAX, fmax( PIV_MIN, 100 * pTol ) );
                    if( new_pTol == pTol ) {
                        /* We have tried all allowed pivot tolerances */
                        LOGMSG("MA27: Failed to factorize matrix\n" );
                        LOGMSG("      Order = %-d, ", ma27->n );
                        if( error == 3 ) {
                            LOGMSG("Rank =");
                        } else {
                            LOGMSG("Singularity at pivot step" );
                        }
                        LOGMSG(" %-d\n", ma27->info[2] );
                        return -2;
                    }
                }
                ma27->cntl[0] = new_pTol;
                if( error == 3 ) finished = 1;  // A factorization was produced!
            }
            else {
                error = Process_Error_Code( ma27, error );
                if( error == -4 ) {
                    /* Must re-initialize factors */
                    cblas_dcopy( ma27->nz, A, 1, ma27->factors, 1 );
                }
                if( error != -3 && error != -4 ) return error;
            }
        }
        LOGMSG( " done\n" );
        return 0;
    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ma27_Solve"
    int Ma27_Solve( Ma27_Data *ma27, double x[] ) {

        LOGMSG( " MA27 :: Solving\n" );

        /* Unpack data structure and call MA27CD */
        MA27CD( &(ma27->n), ma27->factors, &(ma27->la), ma27->iw,
                &(ma27->liw), ma27->w, &(ma27->maxfrt), x,
                ma27->iw1, &(ma27->nsteps), ma27->icntl, ma27->info );

        return ma27->info[0];
    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ma27_Refine"
    int Ma27_Refine( Ma27_Data *ma27, double x[], double rhs[],
                     double A[], double tol, int maxitref ) {

        int    n = ma27->n, i, j, k, nitref;
        double b_norm, resid_norm;

        LOGMSG( " MA27 :: Performing iterative refinement...\n" );

        /* Compute initial residual */
        b_norm = cblas_dnrm_infty( n, rhs, 1 );        
        cblas_dcopy( n, rhs, 1, ma27->residual, 1 );
        for( k = 0; k < ma27->nz; k++ ) {
            i = ma27->irn[k] - 1;
            j = ma27->icn[k] - 1;
            ma27->residual[i] -= A[k] * x[j];
            if( i != j ) ma27->residual[j] -= A[k] * x[i];
        }
        resid_norm = cblas_dnrm_infty( n, ma27->residual, 1 );

        LOGMSG( " Norm of residual: %-g\n", resid_norm );

        /* Perform iterative refinements, if required */
        nitref = 0;
        while( nitref < maxitref && resid_norm > tol * (1+b_norm) ) {

            nitref++;

            /* Solve system again with residual as rhs */
            cblas_dcopy( n, ma27->residual, 1, rhs, 1 );
            MA27CD( &n, ma27->factors, &(ma27->la), ma27->iw,
                    &(ma27->liw), ma27->w, &(ma27->maxfrt),
                    rhs, ma27->iw1, &(ma27->nsteps),
                    ma27->icntl, ma27->info );

            /* Update solution: x <- x + rhs */
            cblas_daxpy( n, 1.0, rhs, 1, x, 1 );
          
            /* Update residual: residual <- residual - A rhs */
            for( k = 0; k < ma27->nz; k++ ) {
                i = ma27->irn[k] - 1;
                j = ma27->icn[k] - 1;
                ma27->residual[i] -= A[k] * rhs[j];
                if( i != j )
                    ma27->residual[j] -= A[k] * rhs[i];
            }
            resid_norm = cblas_dnrm_infty( n, ma27->residual, 1 );
            
            LOGMSG( " Ref %-d: Norm of residual: %-g\n", nitref,
                                                         resid_norm );
        }

        LOGMSG( " done\n" );
        return ma27->info[0];

    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Ma27_Finalize"
    void Ma27_Finalize( Ma27_Data *ma27 ) {

        /* Free allocated memory */
        LOGMSG( " MA27 :: Deallocating data arrays..." );
        NLPy_Free( ma27->irn );
        NLPy_Free( ma27->icn );
        NLPy_Free( ma27->iw  );
        NLPy_Free( ma27->ikeep );
        NLPy_Free( ma27->iw1 );
        NLPy_Free( ma27->factors );
        // Residual is not allocated in NLPy --- it comes from Python
        //NLPy_Free( ma27->residual );
        NLPy_Free( ma27->w );
        NLPy_Free( ma27 );
        return;
    }

/* ================================================================= */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "Process_Error_Code"
    int Process_Error_Code( Ma27_Data *ma27, int nerror ) {

        int newsize;

        LOGMSG( "    [nerror %-2d]", nerror );
        /* Take appropriate action according to error code */
        /* Inflation factors are from the MA27 spec sheet  */
        switch( nerror ) {
            /* Warnings */
            case 1:
                LOGMSG( " Found %d indices out of range\n", ma27->info[1] );
                break;

            case 2:
                LOGMSG( " Found %d sign changes in definite matrix\n",
                        ma27->info[1] );
                break;

            case 3:
                LOGMSG( " Matrix has rank %d, deficient\n", ma27->info[1] );
                break;

            /* Errors */
            case -1:
                LOGMSG( " Value of n is out of range: %d\n", ma27->n );
                break;

            case -2:
                LOGMSG( " Value of nz is out of range: %d\n", ma27->nz );
                break;

            case -3:
                ma27->liw = ceil( 1.2 * ma27->info[1] );
                LOGMSG( " Adjusting size of array IW to %d\n", ma27->liw );
                NLPy_Free( ma27->iw ); ma27->iw = NULL;
                ma27->iw = (int *)NLPy_Calloc( ma27->liw, sizeof(int) );
                break;

            case -4:
                newsize = ceil( 1.2 * ma27->info[1] );
                LOGMSG( " Adjusting size of array FACTORS to %d\n", newsize );
                NLPy_Free( ma27->factors );
                ma27->factors = (double*)NLPy_Calloc( newsize, sizeof(double) );
                ma27->la = newsize;
                break;

            case -5:
                LOGMSG( " Matrix singularity detected at pivot step %d\n",
                        ma27->info[1] );
                break;

            case -6:
                LOGMSG( " Change in pivot sign detected at pivot step %d\n",
                        ma27->info[1] );
                break;

            case -7:
                LOGMSG( " Value of nsteps out of range: %d\n", ma27->nsteps );
                break;

            default:
                LOGMSG( " Unrecognized flag from Factorize()" );
                nerror = -30;
        }    
        return nerror;
    }

/* ================================================================= */

#ifdef __cplusplus
}              /* Closing brace for  extern "C"  block */
#endif
