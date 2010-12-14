#include "amplutils.h"

int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b) {
  return (int)sphsetup(no, ow, y, b);
}
double ampl_objval(ASL* asl, int np, double x[], int* ne) {
  return objval(np, x, ne);
}
void ampl_objgrd(ASL* asl, int np, double x[], double g[], int* ne) {
  objgrd(np, x, g, ne);
}
void ampl_conval(ASL* asl, double x[], double r[], int* ne) {
  conval(x, r, ne);
}
