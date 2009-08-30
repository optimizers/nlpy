      PROGRAM BenchmarkMa27

      IMPLICIT NONE

      INTEGER          NMAX,NNZMAX,LIW
      PARAMETER       (NMAX=12000)
      PARAMETER       (NNZMAX=2000000)
      PARAMETER       (LIW=2500000)
      CHARACTER*256    FILENAME
      CHARACTER*10     REP
      CHARACTER*7      FIELD
      CHARACTER*19     SYMM
      INTEGER          ROWS,COLS,NNZ,I,J,K,NSTEPS,IFLAG,MAXFRT
      DOUBLE PRECISION OPS
      INTEGER          INDX(NNZMAX),JNDX(NNZMAX),IVAL(1)
      INTEGER          ICNTL(30),INFO(20),IW(LIW),IKEEP(3*NMAX)
      INTEGER          IW1(2*NMAX)
      DOUBLE PRECISION RVAL(NNZMAX),W(NMAX),RHS(NMAX),CNTL(5)
      REAL             TANALYZE(2),TSOLVE(2),TTOT
      COMPLEX          CVAL(1)
      LOGICAL          DEBUG
C      EXTERNAL         DTIME
C
C     Initialize
C
      DEBUG=.FALSE.
      IF (DEBUG) THEN
         WRITE(6,*) 'Solutions should be all ones'
      ENDIF
C      WRITE(6,'(A15,A15)') 'Problem','Time'
 1    CONTINUE
      ROWS = 0
      COLS = 0
      NNZ  = 0
C
C     Read file name
C
      READ(5,'(A)',END=900) FILENAME
C
C     Read current test problem
C
      OPEN(50,FILE=FILENAME,FORM='FORMATTED',STATUS='OLD')
      CALL MMINFO(50,REP,FIELD,SYMM,ROWS,COLS,NNZ)
      CLOSE(50)
C
C     Check problem data
C
      IF (ROWS .NE. COLS) THEN
         WRITE(6,'(A)') 'System must be square. Aborting'
         STOP
      ENDIF
      IF (NNZ .GT. NNZMAX) THEN
         WRITE(6,999) NNZ
         STOP
      ENDIF
      IF (ROWS .GT. NMAX) THEN
         WRITE(6,998) ROWS
         STOP
      ENDIF
      IF (SYMM .NE. 'symmetric') THEN
         WRITE(6,'(A)') 'System must be symmetric. Aborting'
         STOP
      ENDIF
      IF (FIELD .NE. 'real') THEN
         WRITE(6,'(A)') 'Matrix entries must be real. Aborting'
         STOP
      ENDIF
C
C     Read problem data
C
      OPEN(50,FILE=FILENAME,FORM='FORMATTED',STATUS='OLD')
      CALL MMREAD(50,REP,FIELD,SYMM,ROWS,COLS,NNZ,NNZMAX,
     .     INDX,JNDX,IVAL,RVAL,CVAL)
      CLOSE(50)
C
C     Print some info if required
C
      IF (DEBUG) THEN
         WRITE(6,'(A)') 'First 10 elements of matrix'
         WRITE(6,'(2I6,D13.3)') (INDX(I),JNDX(I),RVAL(I),I=1,10)
      ENDIF
C
C     Build right-hand side   rhs = A * e
C
      DO 100 I=1,ROWS
         RHS(I)=0.0D+0
 100  CONTINUE
      DO 101 K=1,NNZ
         I=INDX(K)
         J=JNDX(K)
         RHS(I)=RHS(I)+RVAL(K)
         IF (I .NE. J) THEN
            RHS(J)=RHS(J)+RVAL(K)
         ENDIF
 101  CONTINUE
C
C     Initialize timer
C
      CALL DTIME(TANALYZE,TTOT)
C
C     Initialize MA27
C
      CALL MA27ID(ICNTL,CNTL)
      ICNTL(1)=6
      ICNTL(2)=6
      ICNTL(3)=0
      IFLAG=0
C
C     Analyze sparsity pattern
C
      CALL MA27AD(ROWS,NNZ,INDX,JNDX,IW,LIW,IKEEP,IW1,NSTEPS,IFLAG,
     .     ICNTL,CNTL,INFO,OPS)
C
C     Factorize matrix
C
      CALL MA27BD(ROWS,NNZ,INDX,JNDX,RVAL,NNZMAX,IW,LIW,IKEEP,NSTEPS,
     .     MAXFRT,IW1,ICNTL,CNTL,INFO)
C
C     Obtain CPU time for analyze phase
C
      CALL DTIME(TANALYZE,TTOT)
      CALL DTIME(TSOLVE,TTOT)
C
C     Solve linear system
C
      CALL MA27CD(ROWS,RVAL,NNZMAX,IW,LIW,W,MAXFRT,RHS,IW1,NSTEPS,
     .     ICNTL,INFO)
C
C     Obtain CPU time for solve phase
C
      CALL DTIME(TSOLVE,TTOT)
C
C     Print some info if required
C
      IF (DEBUG) THEN
         WRITE(6,'(A)') 'First 10 elements of solution'
         WRITE(6,'(I6,D13.3)') (I,RHS(I),I=1,10)
         WRITE(6,'(A)') 'Last 10 elements of solution'
         WRITE(6,'(I6,D13.3)') (I,RHS(I),I=ROWS-10,ROWS)
      ENDIF
C
C     Print statistics
C
      WRITE(6,'(A15,F15.3,F15.3)')
     .     FILENAME,TANALYZE(1)+TANALYZE(2),TSOLVE(1)+TSOLVE(2)
C
C     Move on to next problem
C
      GOTO 1
C
C     Reached end of problem list
C
 900  CONTINUE
C
C     Non-executable statements
C
 998  FORMAT('BenchmarkMa27: Increase parameter NMAX to at least ',I6)
 999  FORMAT('BenchmarkMa27: Increase parameter NNZMAX to at least ',I6)

      END PROGRAM
