#include "amplutils.h"

int ampl_sphsetup(ASL* asl, int no, int ow, int y, int b) {
  return (int)sphsetup(no, ow, y, b);
}
double ampl_objval(ASL* asl, int np, double x[], int* ne) {
  return objval(np, x, ne);
}
int ampl_objgrd(ASL* asl, int np, double x[], double g[]) {
  int nerror;
  objgrd(np, x, g, &nerror);
  return nerror;
}
int ampl_conval(ASL* asl, double x[], double r[]) {
  int nerror;
  conval(x, r, &nerror);
  return nerror;
}
int ampl_jacval(ASL* asl, double x[], double j[]) {
  int nerror;
  jacval(x, j, &nerror);
  return nerror;
}
int ampl_conival(ASL* asl, int i, double* x, double* c) {
  int nerror;
  *c = conival(i, x, &nerror);
  return nerror;
}
int ampl_congrd(ASL* asl, int i, double* x, double* g) {
  int nerror;
  congrd(i, x, g, &nerror);
  return nerror;
}
void ampl_sphes(ASL* asl, double* H, int nobj, double* ow, double* y) {
  sphes(H, nobj, ow, y);
}
void ampl_hvcomp(ASL* asl, double* hv, double* v, int nobj, double* ow, double* y) {
  hvpinit_ASL(asl, ihd_limit, nobj, ow, y);
  hvcomp(hv, v, nobj, ow, y);
}
void ampl_xknown(ASL* asl, double* x) {
  xknown(x);
}
