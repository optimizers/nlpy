model;
param n := 5;
var x {1..n};

minimize obj:
  sum {i in 1..n-1} (100*(x[i+1] - x[i]^2)^2 + (1-x[i])^2);

let {i in 1..n} x[i] := -1;
