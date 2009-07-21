
/*
 * ==========================================
 * Main header file for the interface between
 * Python and the Ampl library.
 *
 * D. Orban@ECE        M. Friedlander@Argonne
 *                          Chicago 2002-2003
 * ==========================================
 */

#include <math.h>

#ifdef __cplusplus
extern "C" {   /* To prevent C++ compilers from mangling symbols */
#endif

/* AIX does not append underscore to Fortran subroutine names */
#ifdef _AIX
#define FUNDERSCORE(a)   a
#else
#define FUNDERSCORE(a)   a##_
#endif

#include "amplpytype.h"

#define MAX(m,n)  ((m)>(n)?(m):(n))
#define MIN(m,n)  ((m)<(n)?(m):(n))

/* Some real constants -- AmplPyReal is defined in galahad.h */
#define ZERO             (AmplPyReal)0.0
#define ONE              (AmplPyReal)1.0
#define TWO              (AmplPyReal)2.0
#define THREE            (AmplPyReal)3.0
#define FIVE             (AmplPyReal)5.0
#define FORTRAN_INFINITY (AmplPyReal)pow( 10, 20 )

/*
 * =================================
 *  T Y P E   D E F I N I T I O N S
 * =================================
 */

/*
 * Define Fortran types for integer and double precision
 * The following choices are from f2c.h
 */

    typedef long int integer;
    typedef long int logical;
#define FALSE (0)     /* Fortran FALSE */
#define TRUE  (1)     /* Fortran  TRUE */

/*
 * Prototypes
 */

    static void Initial_Point( AmplPyReal *x, AmplPyReal *bl, AmplPyReal *bu, 
			       logical *equatn, logical *linear, AmplPyReal *y,
			       AmplPyReal *cl, AmplPyReal *cu,
			       logical efirst, logical lfirst, logical nvfrst );

    static int  Get_nnzj( void );
    static int  Get_nnzh( void );
    static void Eval_fc( AmplPyReal *x, AmplPyReal f, AmplPyReal *c );
    static void Eval_gJ( logical grlagf, AmplPyReal *v, AmplPyReal *x, AmplPyReal *cjac,
			 AmplPyReal *g, integer *indvar, integer *indfun );
    static void Eval_h( AmplPyReal *x, AmplPyReal *v, AmplPyReal *h,
			integer *irnh, integer *icnh );

/*
 * Define shortcuts for the CUTEr library functions,
 * and try to avoid the trailing underscore.
 *
 */

#define USETUP   FUNDERSCORE(usetup)
#define CSETUP   FUNDERSCORE(csetup)

#define UDIMEN   FUNDERSCORE(udimen)
#define UDIMSH   FUNDERSCORE(udimsh)
#define UDIMSE   FUNDERSCORE(udimse)
#define UVARTY   FUNDERSCORE(uvarty)
#define UNAMES   FUNDERSCORE(unames)
#define UREPRT   FUNDERSCORE(ureprt)

#define CDIMEN   FUNDERSCORE(cdimen)
#define CDIMSJ   FUNDERSCORE(cdimsj)
#define CDIMSH   FUNDERSCORE(cdimsh)
#define CDIMSE   FUNDERSCORE(cdimse)
#define CVARTY   FUNDERSCORE(cvarty)
#define CNAMES   FUNDERSCORE(cnames)
#define CREPRT   FUNDERSCORE(creprt)

#define UFN      FUNDERSCORE(ufn)
#define UGR      FUNDERSCORE(ugr)
#define UOFG     FUNDERSCORE(uofg)
#define UBANDH   FUNDERSCORE(ubandh)
#define UDH      FUNDERSCORE(udh)
#define USH      FUNDERSCORE(ush)
#define UEH      FUNDERSCORE(ueh)
#define UGRDH    FUNDERSCORE(ugrdh)
#define UGRSH    FUNDERSCORE(ugrsh)
#define UGREH    FUNDERSCORE(ugreh)
#define UPROD    FUNDERSCORE(uprod)

#define CFN      FUNDERSCORE(cfn)
#define COFG     FUNDERSCORE(cofg)
#define CCFG     FUNDERSCORE(ccfg)
#define CGR      FUNDERSCORE(cgr)
#define CSGR     FUNDERSCORE(csgr)
#define CCFSG    FUNDERSCORE(ccfsg)
#define CCIFG    FUNDERSCORE(ccifg)
#define CCIFSG   FUNDERSCORE(ccifsg)
#define CGRDH    FUNDERSCORE(cgrdh)
#define CDH      FUNDERSCORE(cdh)
#define CSH      FUNDERSCORE(csh)
#define CEH      FUNDERSCORE(ceh)
#define CIDH     FUNDERSCORE(cidh)
#define CISH     FUNDERSCORE(cish)
#define CSGRSH   FUNDERSCORE(csgrsh)
#define CSGREH   FUNDERSCORE(csgreh)
#define CPROD    FUNDERSCORE(cprod)

#define ELFUN    FUNDERSCORE(elfun)
#define RANGE    FUNDERSCORE(range)
#define GROUP    FUNDERSCORE(group)

#define FORTRAN_OPEN  FUNDERSCORE(fortran_open)
#define FORTRAN_CLOSE FUNDERSCORE(fortran_close)

/*
 * Prototypes for CUTEr FORTRAN routines found in libcuter.a
 * See http://cuter.rl.ac.uk/cuter-www/
 */

/* Setup routines */
    void USETUP( integer *funit, integer *iout, integer *n, AmplPyReal *x, 
		 AmplPyReal *bl, AmplPyReal *bu, integer *nmax );

    void CSETUP( integer *funit, integer *iout, integer *n, integer *m,
		 AmplPyReal *x, AmplPyReal *bl, AmplPyReal *bu, integer *nmax,
		 logical *equatn, logical *linear, AmplPyReal *v, AmplPyReal *cl,
		 AmplPyReal *cu, integer *mmax, logical *efirst,
		 logical *lfirst, logical *nvfrst );

/* Unconstrained dimensioning and report routines */
    void UDIMEN( integer funit, integer *n );
    void UDIMSH( integer *nnzh );
    void UDIMSE( integer *ne, integer *nzh, integer *nzirnh );
    void UVARTY( integer *n, integer *ivarty );
    void UNAMES( integer *n, char *pname, char *vnames );
    void UREPRT( AmplPyReal *calls, AmplPyReal *time );

/* Constrained dimensioning and report routines */
    void CDIMEN( integer *funit, integer *n, integer *m );
    void CDIMSJ( integer *nnzj );
    void CDIMSH( integer *nnzh );
    void CDIMSE( integer *ne, integer *nzh, integer *nzirnh );
    void CVARTY( integer *n, integer *ivarty );
    void CNAMES( integer *n, integer *m, char *pname, char *vnames,
		 char *gnames );
    void CREPRT( AmplPyReal *calls, AmplPyReal *time );


/* Unconstrained optimization routines */
    void UFN( integer *n, AmplPyReal *x, AmplPyReal *f );
    void UGR( integer *n, AmplPyReal *x, AmplPyReal *g );
    void UOFG( integer *n, AmplPyReal *x, AmplPyReal *f, AmplPyReal *g,
	       logical *grad );

    void UBANDH( integer *n, logical *goth, AmplPyReal *x, integer *nsemib,
		 AmplPyReal *bandh, integer *lbandh );

    void UDH( integer *n, AmplPyReal *x, integer *lh1, AmplPyReal *h );
    void USH( integer *n, AmplPyReal *x, integer *nnzh, integer *lh,
	      AmplPyReal *h, integer *irnh, integer *icnh );
    void UEH( integer *n, AmplPyReal *x, integer *ne, integer *irnhi,
	      integer *lirnhi, integer *le, integer *iprnhi, AmplPyReal *hi,
	      integer *lhi, integer *iprhi, logical *byrows );

    void UGRDH( integer *n, AmplPyReal *x, AmplPyReal *g, integer *lh1,
		AmplPyReal *h);
    void UGRSH( integer *n, AmplPyReal *x, AmplPyReal *g, integer *nnzh,
		integer *lh, AmplPyReal *h, integer *irnh, integer *icnh );
    void UGREH( integer *n, AmplPyReal *x, AmplPyReal *g, integer *ne,
		integer *irhni, integer *lirnhi, integer *le, integer *iprnhi,
		AmplPyReal *hi, integer *lhi, integer *iprhi, logical *byrows );

    void UPROD( integer *n, logical *goth, AmplPyReal *x, AmplPyReal *p,
		AmplPyReal *q );

/* Constrained optimization routines */
    void CFN(  integer *n, integer *m, AmplPyReal *x, AmplPyReal *f, integer *lc,
	       AmplPyReal *c );
    void COFG( integer *n, AmplPyReal *x, AmplPyReal *f, AmplPyReal *g,
	       logical *grad );

    void CCFG( integer *n, integer *m, AmplPyReal *x, integer *lc,
	       AmplPyReal *c, logical *jtrans, integer *lcjac1, integer *lcjac2,
	       AmplPyReal *cjac, logical *grad );
    void CGR(  integer *n, integer *m, AmplPyReal *x, logical *grlagf,
	       integer *lv, AmplPyReal *v, AmplPyReal *g, logical *jtrans,
	       integer *lcjac1, integer *lcjac2, AmplPyReal *cjac );
    void CSGR( integer *n, integer *m, logical *grlagf, integer *lv,
	       AmplPyReal *v, AmplPyReal *x, integer *nnzj, integer *lcjac,
	       AmplPyReal *cjac, integer *indvar, integer *indfun );

    void CCFSG(  integer *n, integer *m, AmplPyReal *x, integer *lc,
		 AmplPyReal *c, integer *nnzj, integer *lcjac,
		 AmplPyReal *cjac, integer *indvar, integer *indfun,
		 logical *grad );
    void CCIFG(  integer *n, integer *i, AmplPyReal *x, AmplPyReal *ci,
		 AmplPyReal *gci, logical *grad );
    void CCIFSG( integer *n, integer *i, AmplPyReal *x, AmplPyReal *ci,
		 integer *nnzsgc, integer *lsgci, AmplPyReal *sgci,
		 integer *ivsgci, logical *grad );

    void CGRDH( integer *n, integer *m, AmplPyReal *x, logical *grlagf,
		integer *lv, AmplPyReal *v, AmplPyReal *g, logical *jtrans,
		integer *lcjac1, integer *lcjac2, AmplPyReal *cjac,
		integer *lh1, AmplPyReal *h );

    void CDH( integer *n, integer *m, AmplPyReal *x, integer *lv, AmplPyReal *v,
	      integer *lh1, AmplPyReal *h );
    void CSH( integer *n, integer *m, AmplPyReal *x, integer *lv, AmplPyReal *v,
	      integer *nnzh, integer *lh, AmplPyReal *h, integer *irnh,
	      integer *icnh );
    void CEH( integer *n, integer *m, AmplPyReal *x, integer *lv, AmplPyReal *v,
	      integer *ne, integer *irnhi, integer *lirhni, integer *le,
	      integer *iprnhi, AmplPyReal *hi, integer *lhi, integer *iprhi,
	      logical *byrows );

    void CIDH( integer *n, AmplPyReal *x, integer *iprob, integer *lh1,
	       AmplPyReal *h );
    void CISH( integer *n, AmplPyReal *x, integer *iprob, integer *nnzh,
	       integer *lh, AmplPyReal *h, integer *irnh, integer *icnh );

    void CSGRSH( integer *n, integer *m, AmplPyReal *x, logical *grlagf,
		 integer *lv, AmplPyReal *v, integer *nnzj, integer *lcjac,
		 AmplPyReal *cjac, integer *indvar, integer *indfun,
		 integer *nnzh, integer *lh, AmplPyReal *h, integer *irnh,
		 integer *icnh );
    void CSGREH( integer *n, integer *m, AmplPyReal *x, logical *grlagf,
		 integer *lv, AmplPyReal *v, integer *nnzj, integer *lcjac,
		 AmplPyReal *cjac, integer *indvar, integer *indfun,
		 integer *ne, integer *irnhi, integer *lirhni, integer *le,
		 integer *iprnhi, AmplPyReal *hi, integer *lhi, integer *iprhi,
		 logical *byrows );

    void CPROD( integer *n, integer *m, logical *goth, AmplPyReal *x,
		integer *lv, AmplPyReal *v, AmplPyReal *p, AmplPyReal *q );

/* For backward compatibility with previous versions of CUTE */
#define CSCFG( n, m, x, lc, c, nnzj, lcjac, cjac, indvar, indfun, grad ) CCFSG( n, m, x, lc, c, nnzj, lcjac, cjac, indvar, indfun, grad )
#define CSCIFG( n, i, x, ci, nnzsgc, lsgci, sgci, ivsgci, grad ) CCIFSG( n, i, x, ci, nnzsgc, lsgci, sgci, ivsgci, grad )

/* FORTRAN auxiliary subroutines to retrieve stream unit numbers */
    void FORTRAN_OPEN(  integer *funit, char *fname, integer *ierr );
    void FORTRAN_CLOSE( integer *funit, integer *ierr );

#ifdef __cplusplus
}   /* To prevent C++ compilers from mangling symbols */
#endif
