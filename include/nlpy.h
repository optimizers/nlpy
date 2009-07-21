/*
 * Common definitions for NLPy interfaces
 * $Id:$
 */

#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define imax(a,b) ((a) > (b) ? (a) : (b))
#define imin(a,b) ((a) < (b) ? (a) : (b))
#define ZERO      0.0

#ifdef _AIX
#define FUNDERSCORE(a) a
#else
#define FUNDERSCORE(a) a##_
#endif

/* Memory allocation routines */

void *NLPy_Malloc( void *object, int length, size_t s );
void *NLPy_Calloc( int length, size_t s );
void *NLPy_Realloc( void *object, int length, size_t s );
void  NLPy_Free_Object( void **object );

/* Aliases */

#ifndef Malloc
#define Malloc(object,length,type)  \
  if(!(object = (type *)NLPy_Malloc((void*)(object),length,sizeof(type)))) \
    return PyErr_NoMemory();
#endif
#ifndef Calloc
#define Calloc(object,length,type)  \
  if(!(object = (type *)NLPy_Calloc((void*)(object),length,sizeof(type))) ) \
    return PyErr_NoMemory();
#endif
#ifndef Realloc
#define Realloc(object,length,type) \
  if(!(object = (type *)NLPy_Realloc((void*)(object),length,sizeof(type)))) \
    return PyErr_NoMemory();
#endif

/* Shortcut to make NLPy_Free_Object a bit more intuitive */
#define NLPy_Free(obj) NLPy_Free_Object( (void*)(&(obj)) )

/* Error messages */

#define ERRQ(errcode,msg) {                                          \
    printf( "Error occured in function %s, file %s at line %d\n",    \
            __FUNCT__, __FILE__, __LINE__ );                         \
    printf( "Error:: Code = %d, Msg :: %s\n", errcode, msg );        \
    exit( errcode );                                                 \
}
