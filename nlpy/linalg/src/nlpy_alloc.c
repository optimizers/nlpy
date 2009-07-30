#include <stdlib.h>
#include <stdio.h>
#include "nlpy.h"

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "NLPy_Malloc"
void *NLPy_Malloc( void *object, int length, size_t s ) {
  object = malloc( length * s );
  if( !object ) ERRQ(-1,"Unable to allocate memory");
  return object;
}

/* -------------------------------------------------- */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "NLPy_Calloc"
void *NLPy_Calloc( int length, size_t s ) {
  void *object = calloc( (size_t)length, s );
  if( !object ) ERRQ(-2,"Unable to allocate pointer");
  return object;
}

/* -------------------------------------------------- */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "NLPy_Realloc"
void *NLPy_Realloc( void *object, int length, size_t s ) {
  object = realloc( object, length * s );
  if( !object ) ERRQ(-3,"Unable to reallocate");
  return object;
}

/* -------------------------------------------------- */

#ifdef  __FUNCT__
#undef  __FUNCT__
#endif
#define __FUNCT__ "NLPy_Free_Object"
void NLPy_Free_Object( void **object ) {
  if( *object ) {
    free( *object );
    *object = NULL;
  }
  return;
}
