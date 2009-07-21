/*
 * ============================================
 * A wrapper around the Harwell subroutine MA57
 * to dynamically allocate and ajust workspace.
 *
 * D. Orban            Montreal, September 2003
 *                               August    2008
 * $Id: ma57.h 83 2008-08-18 00:59:53Z d-orban $
 * ============================================
 */

#include "nlpy.h"

#define LOGMSG(...) if (ma57->logfile) fprintf(ma57->logfile, __VA_ARGS__)

#define MA57ID    FUNDERSCORE(ma57id)
#define MA57AD    FUNDERSCORE(ma57ad)
#define MA57BD    FUNDERSCORE(ma57bd)
#define MA57CD    FUNDERSCORE(ma57cd)
#define MA57DD    FUNDERSCORE(ma57dd)
#define MA57ED    FUNDERSCORE(ma57ed)

typedef struct Ma57_Data {
  int       n, nz;               /* Order and #nonzeros */
  int       icntl[20], info[40];
  double    cntl[5];
  double    rinfo[20];
  int      *irn, *jcn;           /* Sparsity pattern    */
  int       lkeep, *keep;        /* Pivot sequence      */
  int      *iwork;               /* Wokspace array      */
  double   *fact;                /* Matrix factors      */
  int       lfact;               /* Size of array fact  */
  int      *ifact;               /* Indexing of factors */
  int       lifact;              /* Size of array ifact */
  int       job;
  int       nrhs;                /* Number of rhs       */
  double   *rhs;                 /* Right-hand sides    */
  int       lrhs;                /* Leading dim of rhs  */
  double   *work;                /* Real workspace      */
  int       lwork;               /* Size of array work  */
  int       calledcd;            /* Flag for MA57DD     */
  double   *x;                   /* Solution to Ax=rhs  */
  double   *residual;            /* = A x - rhs         */
  char      fetched;             /* Factors were fetched
                                  * Used for de-allocation
                                  */
  int       rank, rankdef;
  FILE     *logfile;             /* File for log output */
} Ma57_Data;

/* Below I indicate arrays of fixed length by specifying it
 * explicitly, e.g. icntl[30], arrays of variable length by
 * specifying it implicitly, e.g. iw[]. The remaining variables
 * are either variables of the type indicated, or arrays whose
 * size is fixed, but depends on the value of other parameters.
 */

extern void MA57ID( double cntl[5], int icntl[20] );
extern void MA57AD( int *n, int *ne, int irn[], int jcn[],
                    int *lkeep, int keep[], int iwork[], int icntl[20],
                    int info[40], double rinfo[20] );
extern void MA57BD( int *n, int *ne, double a[], double fact[], int *lfact,
                    int ifact[], int *lifact, int *lkeep, int keep[],
                    int iwork[], int icntl[20], double cntl[5],
                    int info[40], double rinfo[20] );
extern void MA57CD( int *job, int *n, double fact[], int *lfact,
                    int ifact[], int *lifact, int *nrhs, double rhs[],
                    int *lrhs, double work[], int *lwork, int iwork[],
                    int icntl[20], int info[40] );
extern void MA57DD( int *job, int *n, int *ne, double a[], int irn[],
                    int jcn[], double fact[], int *lfact, int ifact[],
                    int *lifact, double rhs[], double x[], double resid[],
                    double work[], int iwork[], int icntl[20], double cntl[5],
                    int info[40], double rinfo[20] );
extern void MA57ED( int *n, int *ic, int keep[], double fact[], int *lfact,
                    double newfac[], int *lnew, int ifact[], int *lifact,
                    int newifc[], int *linew, int info[40] );

/* Interfaces to the above MA57 subroutines */

Ma57_Data *Ma57_Initialize( int nz, int n, FILE *logfile );
int  Ma57_Analyze( Ma57_Data *ma57 );
int  Ma57_Factorize( Ma57_Data *ma57, double A[] );
int  Ma57_Solve( Ma57_Data *ma57, double x[] );
int  Ma57_Refine( Ma57_Data *ma57, double x[], double rhs[], double A[],
                  int maxitref, int job );
void Ma57_Finalize(      Ma57_Data *ma57 );
int  Process_Error_Code( Ma57_Data *ma57, int nerror );

#define LFACT_GROW  1.2
#define LIFACT_GROW 1.2
