var x {1..2};

minimize obj:
  log(1+x[1]^2) - x[2] ;

subject to constr: (1+x[1]^2)^2 + x[2]^2 - 4 = 0;

let x[1] := 2;
let x[2] := 2;
