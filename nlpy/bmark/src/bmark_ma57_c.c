/* Benchmark MA57 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ma57.h"
#include "mmio.h"

int main( int argc, char **argv ) {

    Ma57_Data  *problem;
    FILE       *fp;
    MM_typecode matcode;
    double     *rhs;
    double     *a;
    int         retcode, error;
    int         i, j, k, l, m, n, nz;

    clock_t     t_analyze = (clock_t)0, t_solve = (clock_t)0;

    /* Check arguments */
    if( argc < 2 ) {
        printf( "Usage: %-s file1 [... fileN]\n", argv[0] );
        exit( -1 );
    }

    /* printf( "%-15s  %-15s  %-15s\n", "Problem", "Analyze", "Solve" ); */

    /* Read each input file in turn */
    for( k = 1; k < argc; k++ ) {
        if( (fp = fopen( argv[k], "r" )) == NULL ) {
            fprintf( stderr, "Unable to open %-s\n", argv[k] );
            continue; // Skip to next problem
        }
        if( mm_read_banner( fp, &matcode ) != 0 ) {
            fprintf( stderr, "Could not process MatrixMarket banner\n" );
            fclose(fp);
            continue; // Skip to next problem
        }
        if( !( mm_is_matrix(matcode) && mm_is_coordinate(matcode) &&
               mm_is_real(matcode)   && mm_is_symmetric(matcode) ) ) {
            fprintf( stderr, "%-s: Cannot process matrices of type %-s\n",
                    argv[k], mm_typecode_to_str(matcode) );
            fclose(fp);
            continue; // Skip to next problem
        }
        retcode = mm_read_mtx_crd_size( fp, &m, &n, &nz );
        if( retcode ) {
            fprintf( stderr, "%-s: Error reading matrix size\n", argv[k] );
            fclose(fp);
            continue; // Skip to next problem
        }
        if( m != n ) {
            fprintf( stderr, "%-s: Matrix must be square", argv[k] );
            fprintf( stderr, " (n = %-d, m = %-d)\n", n, m );
            fclose(fp);
            continue; // Skip to next problem
        }
        
        /* Initialize */
        problem = Ma57_Initialize( nz, n, NULL );
        a = (double *)calloc( problem->nz, sizeof(double) );
        if( a == NULL ) {
            fprintf( stderr, "Memory allocation error. Skipping.\n" );
            continue;
        }

        /* Read in matrix entries */
        for( i = 0; i < problem->nz; i++ )
            fscanf( fp, "%d %d %lg\n",
                    &(problem->irn[i]), &(problem->jcn[i]), &(a[i]) );

        fclose(fp);

        /* Allocate memory for right-hand side */
        rhs = (double *)calloc( problem->n,  sizeof(double) );
        if( rhs == NULL ) {
            fprintf( stderr, "Memory allocation error. Skipping.\n" );
            free( a );
            continue;
        }
        for( l = 0; l < problem->n; l++ ) rhs[l] = ZERO;

        /* rhs = A * e. Indices are in Fortran 1-based scheme */
        for( l = 0; l < problem->nz; l++ ) {
            i = problem->irn[l];
            j = problem->jcn[l];
            rhs[i-1] += a[l];
            if( i != j ) rhs[j-1] += a[l];
        }

        t_analyze = clock();

        /* Analyze */
        error = Ma57_Analyze( problem );   // Automatic pivot sequence
        if( error ) {
            fprintf( stderr, "Error return from Ma57_Analyze: %-d\n", error );
            continue;
        }

        /* Factorize */
        error = Ma57_Factorize( problem, a );
        if( error ) {
            fprintf( stderr, "Error return from Ma57_Factorize: %-d\n", error );
            continue;
        }

        t_analyze = clock() - t_analyze;
        t_solve = clock();

        /* Solve linear system */
        error = Ma57_Solve( problem, rhs );
        if( error ) {
            fprintf( stderr, "Error return from Ma57_Solve: %-d\n", error );
            continue;
        }

        t_solve = clock() - t_solve;

        printf( "%-15s  %-15f  %-15f\n",
                argv[k], (double)t_analyze/(double)CLOCKS_PER_SEC,
                (double)t_solve/(double)CLOCKS_PER_SEC );

        /* Free dynamically-allocated memory */
        Ma57_Finalize( problem );
        free( a );
        free( rhs );
    }
}
