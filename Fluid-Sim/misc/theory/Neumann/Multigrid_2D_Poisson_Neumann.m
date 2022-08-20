clear -f
source "2D_Functions.m"

% solve Au = f where A is -laplacian
% (Poisson Equation)
% Neumann Boundary Condition: u' at border is 0
% just note that octave is 1 indexed

% how many segments
n = 8

% values are stored at start and end of each segment (thats why total size is +1)
f = zeros(n+1, n+1);
% using this as a specific example since I know this has a solution!
% with neumann BC theres no guarantee that a solution exists in general!!!
for i=1:n+1
  for j=1:n+1
    x = (j-1)/n;
    y = (i-1)/n;
    f(i,j) = 2*x*y*y - y*y + y - 2*x*y + 2*y*x*x - 2*y*x + x - x*x;
  endfor
endfor

% update / calculate exact solution if it doesnt exist / is incorrect
recalc = false;
if (exist("exactSolution","var") == 1 && size(exactSolution) == size(f))
  f2 = negativeLaplace(exactSolution);
  d = abs(f2 - f);
  if(max(max(d)) > 1e-10)
    recalc = true;
  endif
else
  recalc = true;
endif

if( recalc )
##if( true )
  printf("Recalculating exact solution\n");
  % solve for u with a max error of 1e-10 between Au_k and f
  u = zeros(rows(f), columns(f));
  u = iterateJacobi(u, f, 1500, 1e-10);
  exactSolution = u;
endif

% initial guess for uh
uh = zeros(n+1, n+1);

for i = 10:10:100
    approx = iterateJacobi(uh, f, i, 1e-10);
    averageError = getErrorMinusConstantOffset(approx, exactSolution);
    printf("Error after %d jacobi iterations: %f\n",i, averageError);
endfor

##b = waitforbuttonpress();
printf("\n");

for i = 5:5:25
  printf("Pre and Post smoothing iterations: %d\n", i);

  % initial guess for uh
  uh = zeros(n+1, n+1);

  uh = iterateJacobi(uh, f, i, 1e-10);

  averageError = getErrorMinusConstantOffset(uh, exactSolution);
  printf("Error after pre smoothing: %f\n", averageError);

  rh = f - negativeLaplace(uh);
  r2h = restrict(rh);
  e2h = zeros(rows(r2h), columns(r2h));
  e2h = iterateJacobi(e2h, r2h, 2*i, 1e-10);

  eh = interpolate(e2h);
  uh = uh + eh;

  averageError = getErrorMinusConstantOffset(uh, exactSolution);
  printf("Error after correction: %f\n", averageError);

  uh = iterateJacobi(uh, f, i, 1e-10);
  averageError = getErrorMinusConstantOffset(uh, exactSolution);
  printf("Error after post smoothing: %f\n", averageError);

  printf("\n");
endfor
