/*
 * ============================================
 * A wrapper around the Harwell subroutine MA27
 * to dynamically allocate and ajust workspace.
 *
 * D. Orban            Montreal, September 2003
 * M. P. Friedlander   Vancouver,  October 2006
 * $Id: ma27.h 83 2008-08-18 00:59:53Z d-orban $
 * ============================================
 */

#include "nlpy.h"

#define LOGMSG(...) if (ma27->logfile) fprintf(ma27->logfile, __VA_ARGS__);

#define MA27ID        FUNDERSCORE(ma27id)
#define MA27AD        FUNDERSCORE(ma27ad)
#define MA27BD        FUNDERSCORE(ma27bd)
#define MA27CD        FUNDERSCORE(ma27cd)
#define MA27FACTORS   FUNDERSCORE(ma27factors)
#define MA27QDEMASC   FUNDERSCORE(ma27qdemasc)

typedef struct Ma27_Data {
    int     n, nz;               /* Order and #nonzeros */
    int     icntl[30], info[20];
    double  cntl[5];
    int    *irn, *icn;           /* Sparsity pattern    */
    int    *iw, liw;             /* Integer workspace   */
    int    *ikeep;               /* Pivot sequence      */
    int    *iw1;                 /* Integer workspace   */
    int     nsteps;
    int     iflag;               /* Pivot selection     */
    double  ops;                 /* Operation count     */

    char    rankdef;             /* Indicate whether matrix is rank-deficient */
    int     rank;                /* Matrix rank         */

    int     la;
    double *factors;             /* Matrix factors      */
    int     maxfrt;
    double *w;                   /* Real workspace      */

    double *residual;            /* = b - Ax            */

    char    fetched;             /* Factors have been fetched
                                  * Used for de-allocation
                                  */
    FILE   *logfile;             /* File for log output */
} Ma27_Data;

/* Below I indicate arrays of fixed length by specifying it
 * explicitly, e.g. icntl[30], arrays of variable length by
 * specifying it implicitly, e.g. iw[]. The remaining variables
 * are either variables of the type indicated, or arrays whose
 * size is fixed, but depends on the value of other parameters.
 */

extern void MA27ID( int icntl[30], double cntl[5] );
extern void MA27AD( int *n, int *nz, int *irn, int *icn, int iw[],
		    int *liw, int *ikeep, int *iw1, int *nsteps,
		    int *iflag, int icntl[30], double cntl[5],
		    int info[20], double *ops );
extern void MA27BD( int *n, int *nz, int *irn, int *icn,
		    double *a, int *la, int iw[], int *liw,
		    int *ikeep, int *nsteps, int *maxfrt, int *iw1,
		    int icntl[30], double cntl[5], int info[20] );
extern void MA27CD( int *n, double *a, int *la, int iw[], int *liw,
		    double *w, int *maxfrt, double *rhs, int *iw1,
		    int *nsteps, int icntl[30], int info[20] );
extern void MA27FACTORS( int *n, double a[], int *la, int iw[],
                         int *liw, int *maxfrt,
                         int iw2[], int *nblk, int *latop,
                         int icntl[30], int colrhs[], int *nnzD,
                         int id[], int jd[], double d[],
                         int *nnzL, int il[], int jl[], double l[] );
extern void MA27QDEMASC( int *n, int iw[], int *liwm1, int iw2[],
                         int *nblk, int *latop, int icntl[30] );


/* Interfaces to the above MA27 subroutines */

Ma27_Data * Ma27_Initialize(    int nz,          int n, FILE *logfile );
int         Ma27_Analyze(       Ma27_Data *data, int iflag  );
int         Ma27_Factorize(     Ma27_Data *data, double A[] );
int         Ma27_Solve(         Ma27_Data *data, double x[] );
int         Ma27_Refine(        Ma27_Data *data, double x[], double rhs[],
                                double A[], double tol, int maxitref );
void        Ma27_Finalize(      Ma27_Data *data             );
int         Process_Error_Code( Ma27_Data *data, int error  );

#define LIW_MIN    500
#define PIV_MIN   -0.5
#define PIV_MAX    0.5
