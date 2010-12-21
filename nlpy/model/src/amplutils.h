#ifndef __AMPLUTILS_H__
#define __AMPLUTILS_H__

#ifndef real
   #define real double
   #define MYHEADER_DEFN 1
#endif

#include "asl_pfgh.h"
#include "getstub.h"
#include "jacpdim.h"

// These macros aren't needed, and in fact get in the way.
#undef n_var
#undef nbv
#undef niv
#undef n_con
#undef n_obj
#undef nlo
#undef nranges
#undef nlc
#undef nlnc
#undef nlvb
#undef nlvbi
#undef nlvc
#undef nlvci
#undef nlvo
#undef nlvoi
#undef lnc
#undef nzc
#undef nzo
#undef maxrownamelen
#undef maxcolnamelen
#undef n_conjac
#undef want_xpi0
#undef objtype
#undef Ograd
#undef X0
#undef pi0
#undef LUv
#undef Uvx
#undef LUrhs
#undef Urhsx

int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b);
double ampl_objval(ASL* asl, int np, double x[], int* ne);
int ampl_objgrd(ASL* asl, int np, double x[], double g[]);
int ampl_conval(ASL* asl, double x[], double r[]);
int ampl_jacval(ASL* asl, double x[], double j[]);
int ampl_conival(ASL* asl, int i, double* c, double* x);
int ampl_congrd(ASL* asl, int i, double* c, double* x);
void ampl_sphes(ASL* asl, double* H, int nobj, double* ow, double* y);
void ampl_hvcomp(ASL* asl, double* hv, double* p, int nobj, double* ow, double* y);

#ifdef MYHEADER_DEFN
   #undef real
   #undef MYHEADER_DEFN
#endif

#endif
