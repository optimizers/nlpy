#ifndef __AMPLUTILS_H__
#define __AMPLUTILS_H__

#ifndef real
   #define real double
   #define MYHEADER_DEFN 1
#endif

#include "asl_pfgh.h"
#include "getstub.h"
#include "jacpdim.h"

int  ampl_get_n_var(ASL*);
int  ampl_get_n_con(ASL*);
void ampl_set_want_xpi0(ASL*, int);
int  ampl_get_objtype(ASL*, int);
int  ampl_get_nnzj(ASL*);
void  ampl_get_dims(ASL*, int*, int*);

#ifdef MYHEADER_DEFN
   #undef real
   #undef MYHEADER_DEFN
#endif

#endif
