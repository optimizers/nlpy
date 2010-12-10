#ifndef __AMPLUTILS_H__
#define __AMPLUTILS_H__

#ifndef real
   #define real double
   #define MYHEADER_DEFN 1
#endif

#include "asl_pfgh.h"
#include "getstub.h"
#include "jacpdim.h"

#define AMPL_GET(x) int ampl_get_##x(ASL* asl)

AMPL_GET(n_var);
AMPL_GET(nbv);
AMPL_GET(niv);
AMPL_GET(n_con);
AMPL_GET(n_obj);
AMPL_GET(nlo);
AMPL_GET(nranges);
AMPL_GET(nlc);
AMPL_GET(nlnc);
AMPL_GET(nlvb);
AMPL_GET(nlvbi);
AMPL_GET(nlvc);
AMPL_GET(nlvci);
AMPL_GET(nlvo);
AMPL_GET(nlvoi);
AMPL_GET(lnc);
AMPL_GET(nzc);
AMPL_GET(nzo);
AMPL_GET(maxrownamelen);
AMPL_GET(maxcolnamelen);

void ampl_set_n_conjac(ASL* asl, int val0, int val1);
void ampl_get_n_conjac(ASL* asl, int *val0, int *val1);
void ampl_set_want_xpi0(ASL* asl, int val);
int ampl_get_objtype(ASL* asl, int nobj);
int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b);

#ifdef MYHEADER_DEFN
   #undef real
   #undef MYHEADER_DEFN
#endif

#endif
