
/*
 * AmplPy-specific common header file
 */

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {   /* To prevent C++ compilers from mangling symbols */
#endif

/*
 * Type definitions
 */

#ifdef SinglePrecision
    typedef float  AmplPyReal;     /* Single precision real numbers */
#define     AmplCast    (double)   /* Used to cast single to double */
#define     RealCast    (float)    /* Used to cast back to single   */
#else
    typedef double AmplPyReal;     /* Double precision real numbers */
#define     RealCast               /* No cast needed as Ampl uses doubles */
#define     AmplCast               /* No cast needed as Ampl uses doubles */
#endif

/*
 * Treatment of error
 */

#define AMPLPY_ERR_ARG_BADPTR   -1
#define NOT_YET_IMPLEMENTED      -2
#define DIMENSION_MISMATCH       -3
#define INPUT_OUTPUT_ERROR       -4
#define AMBIGUOUS_SOLVER_NAME    -5
#define ELFUN_UNDEFINED          -6
#define GROUP_UNDEFINED          -7

#define SETERRQ(n,s) {                                     \
  fprintf( stderr, "  Amplpy Error::      Code : %d\n", n );    \
  fprintf( stderr, "                       Msg :: %s\n", s );    \
  fprintf( stderr, "  Error  occured in  function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
  exit( n );                                                        \
}

#define SETERRQi(n,s,i) {                                  \
  fprintf( stderr, "  Amplpy Error::     Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s );    \
  fprintf( stderr, "                      Value : %d\n", i );    \
  fprintf( stderr, "  Error  occured in  function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
  exit( n );                                                        \
}


#define SETWARNQ(n,s) {                                      \
  fprintf( stderr, "  Amplpy Warning::    Code : %d\n", n );    \
  fprintf( stderr, "                       Msg :: %s\n", s );    \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define SETWARNQi(n,s,i) {                                   \
  fprintf( stderr, "  Amplpy Warning::   Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s );    \
  fprintf( stderr, "                      Value : %d\n", i );    \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define SETWARNQs(n,s1,s2) {                                 \
  fprintf( stderr, "  Amplpy Warning::   Code  : %d\n", n );    \
  fprintf( stderr, "                      Msg  :: %s\n", s1 );   \
  fprintf( stderr, "                      Value : %s\n", s2 );   \
  fprintf( stderr, "  Warning occured in function %s\n", __FUNCT__ ); \
  fprintf( stderr, "                        file: %s\n", __FILE__ );  \
  fprintf( stderr, "                        line: %d\n", __LINE__ );  \
  fprintf( stderr, " -------------------------------\n" );            \
}

#define AmplpyValidPointer(h)                                        \
  {if (!h) {SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                           \
    SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Invalid Pointer");                \
  }}

#define AmplpyValidCharPointer(h)                                    \
  {if (!h) {SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Null Pointer");}          \
  }

#define AmplpyValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                           \
    SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Invalid Pointer to Int");         \
  }}

#define AmplpyValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3) {                          \
    SETERRQ(AMPLPY_ERR_ARG_BADPTR,"Invalid Pointer to Scalar");      \
  }}


#ifdef __cplusplus
}    /* To prevent C++ compilers from mangling symbols */
#endif
