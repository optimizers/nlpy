C     $Revision: 17 $
C     $Date: 2005-08-12 18:15:51 -0400 (Fri, 12 Aug 2005) $
C     Subroutine MA27FACTORS retrieves the factor L and the inverse
C     of D in the factorization P^T A P = L D L^T performed by MA27,
C     where P is a permutation matrix, A is the matrix factorized
C     in MA27BD, L is unit upper triangular and D is block diagonal with
C     1x1 and 2x2 blocks. L and the lower triangle of D^{-1} are
C     returned in sparse coordinate format.
C
C     M. Friedlander and D. Orban, Montreal May 2005.
C
      SUBROUTINE MA27FACTORS(N, A, LA, IW, LIW, MAXFRT, IW2, NBLK, 
     &                       LATOP, ICNTL, colrhs,
     &                       nnzD, iD, jD, D,
     &                       nnzL, iL, jL, L )
C THIS SUBROUTINE PERFORMS BACKWARD ELIMINATION OPERATIONS
C     USING THE FACTORS DINVERSE AND U
C     STORED IN A/IW AFTER MA27B/BD.
C
C N      - MUST BE SET TO THE ORDER OF THE MATRIX. NOT ALTERED
C          BY MA27R/RD.
C A      - MUST BE SET TO HOLD THE REAL VALUES CORRESPONDING
C          TO THE FACTORS OF DINVERSE AND U.  THIS MUST BE
C          UNCHANGED SINCE THE PRECEDING CALL TO MA27B/BD.  NOT ALTERED
C          BY MA27R/RD.
C LA     - LENGTH OF ARRAY A. NOT ALTERED BY MA27R/RD.
C IW     - HOLDS THE INTEGER INDEXING
C          INFORMATION FOR THE MATRIX FACTORS IN A.  THIS MUST BE
C          UNCHANGED SINCE THE PRECEDING CALL TO MA27B/BD.  NOT ALTERED
C          BY MA27R/RD.
C LIW    - LENGTH OF ARRAY IW.  NOT ALTERED BY MA27R/RD.
C MAXFRT - INTEGER VARIABLE.  MUST BE SET TO THE LARGEST NUMBER OF
C          VARIABLES IN ANY BLOCK PIVOT ROW.  THIS VALUE WAS GIVEN
C          ON OUTPUT FROM MA27B/BD.  NOT ALTERED BY MA27R/RD.
C IW2    - ON ENTRY IW2(I) (I = 1,NBLK)
C          MUST HOLD POINTERS TO THE
C          BEGINNING OF EACH BLOCK PIVOT IN ARRAY IW, AS SET BY
C          MA27Q/QD.
C NBLK   - NUMBER OF BLOCK PIVOT ROWS. NOT ALTERED BY MA27R/RD.
C LATOP  - IT IS THE POSITION IN
C          A OF THE LAST ENTRY IN THE  FACTORS. IT MUST BE UNCHANGED
C          SINCE THE CALL TO MA27Q/QD.  IT IS NOT ALTERED BY MA27R/RD.
C ICNTL is an INTEGER array of length 30, see MA27A/AD.
C   ICNTL(IFRLVL+I) I=1,20 IS USED TO CONTROL WHETHER DIRECT OR INDIRECT
C     ACCESS IS USED BY MA27C/CD.  INDIRECT ACCESS IS EMPLOYED
C     IN FORWARD AND BACK SUBSTITUTION RESPECTIVELY IF THE SIZE OF
C     A BLOCK IS LESS THAN ICNTL(IFRLVL+MIN(10,NPIV)) AND
C     ICNTL(IFRLVL+10+MIN(10,NPIV)) RESPECTIVELY, WHERE NPIV IS THE
C     NUMBER OF PIVOTS IN THE BLOCK.
C
      IMPLICIT NONE
      INTEGER IFRLVL
      PARAMETER ( IFRLVL=5 )
C
C     .. Scalar Arguments ..
      INTEGER LA,LATOP,LIW,MAXFRT,N,NBLK
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IW(LIW),IW2(NBLK),ICNTL(30)
      integer          colrhs(maxfrt)
C     ..
C     .. Local Scalars ..
      INTEGER APOS,APOS2,I1RHS,I2RHS,IBLK,IFR,IIPIV,IIRHS,ILVL,IPIV,
     +        IPOS,IRHS,IST,J,J1,J2,JJ,JJ1,JJ2,JPIV,JPOS,K,LIELL,LOOP,
     +        NPIV
      integer          nnzD, nnzL, pD, pL
      integer          iD(nnzD), jd(nnzD), iL(nnzL), jL(nnzL)
      double precision  D(nnzD),            L(nnzl)
      
      double precision  one
      parameter        (one = 1.0d+0)

C     ..
C     .. Intrinsic Functions ..
      INTRINSIC ABS,MIN
C     ..
C     .. Executable Statements ..

       DO 1 pL = 1, N
          iL(pL) = pL
          jL(pL) = pL
           L(pL) = one
   1   CONTINUE
       pD = 1
       pL = N+1

C APOS. RUNNING POINTER TO CURRENT PIVOT POSITION IN ARRAY A.
C IPOS. RUNNING POINTER TO BEGINNING OF CURRENT BLOCK PIVOT ROW.
      APOS = LATOP + 1
      NPIV = 0
      IBLK = NBLK + 1
C RUN THROUGH BLOCK PIVOT ROWS IN THE REVERSE ORDER.
      DO 180 LOOP = 1,N
        IF (NPIV.GT.0) GO TO 110
        IBLK = IBLK - 1
        IF (IBLK.LT.1) GO TO 190
        IPOS = IW2(IBLK)
C ABS(LIELL) IS NUMBER OF VARIABLES (COLUMNS) IN BLOCK PIVOT ROW.
        LIELL = -IW(IPOS)
C NPIV IS NUMBER OF PIVOTS (ROWS) IN BLOCK PIVOT.
        NPIV = 1
        IF (LIELL.GT.0) GO TO 10
        LIELL = -LIELL
        IPOS = IPOS + 1
        NPIV = IW(IPOS)
   10   JPOS = IPOS + NPIV
        J2 = IPOS + LIELL
        ILVL = MIN(10,NPIV) + 10
        IF (LIELL.LT.ICNTL(IFRLVL+ILVL)) GO TO 110
C
C PERFORM OPERATIONS USING DIRECT ADDRESSING.
C
        J1 = IPOS + 1
C LOAD APPROPRIATE COMPONENTS OF RHS INTO W.
        IFR = 0
        DO 20 JJ = J1,J2
          J = ABS(IW(JJ)+0)
          IFR = IFR + 1
          COLRHS(IFR)=J
CC          W(IFR) = RHS(J)
   20   CONTINUE
C JPIV IS USED AS A FLAG SO THAT IPIV IS INCREMENTED CORRECTLY AFTER
C     THE USE OF A 2 BY 2 PIVOT.
        JPIV = 1
C PERFORM ELIMINATIONS.
        DO 90 IIPIV = 1,NPIV
          JPIV = JPIV - 1
          IF (JPIV.EQ.1) GO TO 90
          IPIV = NPIV - IIPIV + 1
          IF (IPIV.EQ.1) GO TO 30
C JUMP IF WE HAVE A 2 BY 2 PIVOT.
          IF (IW(JPOS-1).LT.0) GO TO 60
C PERFORM BACK-SUBSTITUTION USING 1 BY 1 PIVOT.
   30     JPIV = 1
          APOS = APOS - (LIELL+1-IPIV)
          IST = IPIV + 1
CC          W1 = W(IPIV)*A(APOS)
          ID(PD) = IPIV
          JD(PD) = IPIV
           D(PD) = A(APOS)
          PD=PD+1
          IF (LIELL.LT.IST) GO TO 50
          JJ1 = APOS + 1
          DO 40 J = IST,LIELL
CC            W1 = W1 + A(JJ1)*W(J)
             iL(pL) = iPiv
             jL(pL) = colrhs(j)
              L(pL) = -A(JJ1)
             pL     = pL  + 1
             JJ1    = JJ1 + 1
   40     CONTINUE
   50     CONTINUE
CC          W(IPIV) = W1
          JPOS = JPOS - 1
          GO TO 90
C PERFORM BACK-SUBSTITUTION OPERATIONS WITH 2 BY 2 PIVOT
   60     JPIV = 2
          APOS2 = APOS - (LIELL+1-IPIV)
          APOS = APOS2 - (LIELL+2-IPIV)
          IST = IPIV + 1
CC          W1 = W(IPIV-1)*A(APOS) + W(IPIV)*A(APOS+1)
CC          W2 = W(IPIV-1)*A(APOS+1) + W(IPIV)*A(APOS2)
          ID(PD) = COLRHS(IPIV-1)
          JD(PD) = COLRHS(IPIV-1)
           D(PD) = A(APOS)
          PD=PD+1
          ID(PD) = COLRHS(IPIV)
          JD(PD) = COLRHS(IPIV-1)
           D(PD) = A(APOS+1)
          PD=PD+1
          ID(PD) = COLRHS(IPIV-1)
          JD(PD) = COLRHS(IPIV)
           D(PD) = A(APOS+1)
          PD=PD+1
          ID(PD) = COLRHS(IPIV)
          JD(PD) = COLRHS(IPIV)
           D(PD) = A(APOS2)
          PD=PD+1
          IF (LIELL.LT.IST) GO TO 80
          JJ1 = APOS + 2
          JJ2 = APOS2 + 1
          DO 70 J = IST,LIELL
CC            W1 = W1 + W(J)*A(JJ1)
CC            W2 = W2 + W(J)*A(JJ2)
            IL(PL) = COLRHS(IPIV-1)
            JL(PL) = COLRHS(J)
             L(PL) = -A(JJ1)
            pL=pL+1
            IL(PL) = COLRHS(IPIV)
            JL(PL) = COLRHS(J)
             L(PL) = -A(JJ2)
            pL=pL+1
            JJ1 = JJ1 + 1
            JJ2 = JJ2 + 1
   70     CONTINUE
          iL(pL) = COLRHS(IPIV-1)
          jL(pL) = COLRHS(IPIV)
           L(pL) = one
          pL = pL + 1
   80     CONTINUE
CC          W(IPIV-1) = W1
CC          W(IPIV) = W2
          JPOS = JPOS - 2
   90   CONTINUE
C RELOAD WORKING VECTOR INTO SOLUTION VECTOR.
        IFR = 0
        DO 100 JJ = J1,J2
          J = ABS(IW(JJ)+0)
          IFR = IFR + 1
CC          RHS(J) = W(IFR)
  100   CONTINUE
        NPIV = 0
        GO TO 180
C
C PERFORM OPERATIONS USING INDIRECT ADDRESSING.
C
  110   IF (NPIV.EQ.1) GO TO 120
C JUMP IF WE HAVE A 2 BY 2 PIVOT.
        IF (IW(JPOS-1).LT.0) GO TO 150
C PERFORM BACK-SUBSTITUTION USING 1 BY 1 PIVOT.
  120   NPIV = NPIV - 1
        APOS = APOS - (J2-JPOS+1)
        IIRHS = IW(JPOS)
CC        W1 = RHS(IIRHS)*A(APOS)
        ID(PD) = IIRHS
        JD(PD) = IIRHS
         D(PD) = A(APOS)
        PD=PD+1
        J1 = JPOS + 1
        IF (J1.GT.J2) GO TO 140
        K = APOS + 1
        DO 130 J = J1,J2
          IRHS = ABS(IW(J)+0)
CC          W1 = W1 + A(K)*RHS(IRHS)
          iL(pL) = IIRHS
          jL(pL) =  IRHS
           L(pL) = -A(K)
          pL     = pL + 1
          K      = K  + 1
  130   CONTINUE
  140   CONTINUE
CC        RHS(IIRHS) = W1
        JPOS = JPOS - 1
        GO TO 180
C PERFORM OPERATIONS WITH 2 BY 2 PIVOT
  150   NPIV = NPIV - 2
        APOS2 = APOS - (J2-JPOS+1)
        APOS = APOS2 - (J2-JPOS+2)
        I1RHS = -IW(JPOS-1)
        I2RHS = IW(JPOS)
CC        W1 = RHS(I1RHS)*A(APOS) + RHS(I2RHS)*A(APOS+1)
CC        W2 = RHS(I1RHS)*A(APOS+1) + RHS(I2RHS)*A(APOS2)
        ID(PD) = I1RHS
        JD(PD) = I1RHS
         D(PD) = A(APOS)
        pD=pD+1
        ID(PD) = I2RHS
        JD(PD) = I1RHS
         D(PD) = A(APOS+1)
        pD=pD+1
        ID(PD) = I1RHS
        JD(PD) = I2RHS
         D(PD) = A(APOS+1)
        pD=pD+1
        ID(PD) = I2RHS
        JD(PD) = I2RHS
         D(PD) = A(APOS2)
        pD=pD+1
        J1 = JPOS + 1
        IF (J1.GT.J2) GO TO 170
        JJ1 = APOS + 2
        JJ2 = APOS2 + 1
        DO 160 J = J1,J2
          IRHS = ABS(IW(J)+0)
CC          W1 = W1 + RHS(IRHS)*A(JJ1)
CC          W2 = W2 + RHS(IRHS)*A(JJ2)
          IL(PL) = I1RHS
          JL(PL) = IRHS
           L(PL) = -A(JJ1)
          pL=pL+1
          IL(PL) = I2RHS
          JL(PL) = IRHS
           L(PL) = -A(JJ2)
          PL  = PL + 1
          JJ1 = JJ1 + 1
          JJ2 = JJ2 + 1
  160   CONTINUE
CCCCCC  Not needed!
  170   CONTINUE
CC        RHS(I1RHS) = W1
CC        RHS(I2RHS) = W2
        JPOS = JPOS - 2
  180 CONTINUE
  190 CONTINUE 
C     Set nnzX to actual nnzX.  They may have been overstated.
      nnzL = pL-1
      nnzD = pD-1
      RETURN
      END


      SUBROUTINE MA27QDEMASC(N, IW, LIW, IW2, NBLK, LATOP, ICNTL)
C THIS SUBROUTINE PERFORMS FORWARD ELIMINATION
C     USING THE FACTOR U TRANSPOSE STORED IN A/IA AFTER MA27B/BD.
C
C N      - MUST BE SET TO THE ORDER OF THE MATRIX. NOT ALTERED
C          BY MA27Q/QD.
C IW     - HOLDS THE INTEGER INDEXING
C          INFORMATION FOR THE MATRIX FACTORS IN A.  THIS MUST BE
C          UNCHANGED SINCE THE PRECEDING CALL TO MA27B/BD.  NOT ALTERED
C          BY MA27Q/QD.
C LIW    - LENGTH OF ARRAY IW.  NOT ALTERED BY MA27Q/QD.
C IW2    - NEED NOT BE SET ON ENTRY. ON EXIT IW2(I) (I = 1,NBLK)
C          WILL HOLD POINTERS TO THE
C          BEGINNING OF EACH BLOCK PIVOT IN ARRAY IW.
C NBLK   - NUMBER OF BLOCK PIVOT ROWS. NOT ALTERED BY MA27Q/QD.
C LATOP  - NEED NOT BE SET ON ENTRY. ON EXIT, IT IS THE POSITION IN
C          A OF THE LAST ENTRY IN THE  FACTORS. IT MUST BE PASSED
C          UNCHANGED TO MA27R/RD.
C ICNTL is an INTEGER array of length 30, see MA27A/AD.
C   ICNTL(IFRLVL+I) I=1,20 IS USED TO CONTROL WHETHER DIRECT OR INDIRECT
C     ACCESS IS USED BY MA27C/CD.  INDIRECT ACCESS IS EMPLOYED
C     IN FORWARD AND BACK SUBSTITUTION RESPECTIVELY IF THE SIZE OF
C     A BLOCK IS LESS THAN ICNTL(IFRLVL+MIN(10,NPIV)) AND
C     ICNTL(IFRLVL+10+MIN(10,NPIV)) RESPECTIVELY, WHERE NPIV IS THE
C     NUMBER OF PIVOTS IN THE BLOCK.
C
      IMPLICIT NONE
      INTEGER IFRLVL
      PARAMETER ( IFRLVL=5 )
C     .. Scalar Arguments ..
      INTEGER LATOP,LIW,N,NBLK
C     ..
C     .. Array Arguments ..
      INTEGER IW(LIW),IW2(NBLK),ICNTL(30)
C     ..
C     .. Local Scalars ..
      INTEGER APOS,IBLK,ILVL,IPIV,IPOS,IROW,IST,J1,J2,J3,
     +        JPIV,LIELL,NPIV
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC ABS,MIN
C     ..
C     .. Executable Statements ..
C APOS. RUNNING POINTER TO CURRENT PIVOT POSITION IN ARRAY A.
C IPOS. RUNNING POINTER TO BEGINNING OF BLOCK PIVOT ROW IN IW.
      APOS = 1
      IPOS = 1
      J2 = 0
      IBLK = 0
      NPIV = 0
      DO 140 IROW = 1,N
        IF (NPIV.GT.0) GO TO 90
        IBLK = IBLK + 1
        IF (IBLK.GT.NBLK) GO TO 150
        IPOS = J2 + 1
C SET UP POINTER IN PREPARATION FOR BACK SUBSTITUTION.
        IW2(IBLK) = IPOS
C ABS(LIELL) IS NUMBER OF VARIABLES (COLUMNS) IN BLOCK PIVOT ROW.
        LIELL = -IW(IPOS)
C NPIV IS NUMBER OF PIVOTS (ROWS) IN BLOCK PIVOT.
        NPIV = 1
        IF (LIELL.GT.0) GO TO 10
        LIELL = -LIELL
        IPOS = IPOS + 1
        NPIV = IW(IPOS)
   10   J1 = IPOS + 1
        J2 = IPOS + LIELL
        ILVL = MIN(NPIV,10)
        IF (LIELL.LT.ICNTL(IFRLVL+ILVL)) GO TO 90
C
C PERFORM OPERATIONS USING DIRECT ADDRESSING.
C
C JPIV IS USED AS A FLAG SO THAT IPIV IS INCREMENTED CORRECTLY AFTER
C THE USE OF A 2 BY 2 PIVOT.
        JPIV = 1
        J3 = J1
C PERFORM OPERATIONS.
        DO 70 IPIV = 1,NPIV
          JPIV = JPIV - 1
          IF (JPIV.EQ.1) GO TO 70
C JUMP IF WE HAVE A 2 BY 2 PIVOT.
          IF (IW(J3).LT.0) GO TO 40
C PERFORM FORWARD SUBSTITUTION USING 1 BY 1 PIVOT.
          JPIV = 1
          J3 = J3 + 1
          APOS = APOS + 1
          IST = IPIV + 1
          IF (LIELL.LT.IST) GO TO 70
          APOS = APOS + LIELL - IST + 1
          GO TO 70
C PERFORM OPERATIONS WITH 2 BY 2 PIVOT.
   40     JPIV = 2
          J3 = J3 + 2
          APOS = APOS + 2
          IST = IPIV + 2
          IF (LIELL.LT.IST) GO TO 60
   60     APOS = APOS + 2* (LIELL-IST+1) + 1
   70   CONTINUE
        NPIV = 0
        GO TO 140
C
C PERFORM OPERATIONS USING INDIRECT ADDRESSING.
C
C JUMP IF WE HAVE A 2 BY 2 PIVOT.
   90   IF (IW(J1).LT.0) GO TO 110
C PERFORM FORWARD SUBSTITUTION USING 1 BY 1 PIVOT.
        NPIV = NPIV - 1
        APOS = APOS + 1
        J1 = J1 + 1
        IF (J1.GT.J2) GO TO 140
        APOS = APOS + J2 - J1 + 1
        GO TO 140
C PERFORM OPERATIONS WITH 2 BY 2 PIVOT
  110   NPIV = NPIV - 2
        J1 = J1 + 2
        APOS = APOS + 2
        IF (J1.GT.J2) GO TO 130
  130   APOS = APOS + 2* (J2-J1+1) + 1
  140 CONTINUE
  150 LATOP = APOS - 1
      RETURN
      END
