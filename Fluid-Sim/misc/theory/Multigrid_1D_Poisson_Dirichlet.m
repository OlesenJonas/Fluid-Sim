source "1D_Functions.m"

% solve Au = f where A is -laplacian
% (Poisson Equation)
% Dirichlet Boundary Condition: u(0) = u(size) = 0
% just note that octave is 1 indexed

% how many segments
n = 16

% values are stored at start and end of each segment (thats why total size is +1)
rand("state", 42);
f = rand(1, n+1);

% solve for u with a max error of 1e-10 between Au_k and f
u = zeros(1, columns(f));
% takes 1159 iterations
u = iterateJacobi(u, f, 5000, 1e-10)
exactSolution = u;


% initial guess for uh
uh = zeros(1, n+1);

for i = 10:10:100
    approx = iterateJacobi(uh, f, i, 1e-10);
##    averageError = meansq(approx - exactSolution);
    averageError = mean(abs(approx - exactSolution));
    printf("Error after %d jacobi iterations: %f\n",i, averageError);
endfor

##b = waitforbuttonpress();
printf("\n");

for i = 5:5:15
  printf("Pre and Post smoothing iterations: %d\n", i);

  % initial guess for uh
  uh = zeros(1, n+1);

  uh = iterateJacobi(uh, f, i, 1e-10);

  averageError = mean(abs(uh - exactSolution));
  printf("Error after pre smoothing: %f\n", averageError);

  rh = f - negativeLaplaceWithoutBorder(uh);
  r2h = restrict(rh);
  e2h = zeros(1, columns(r2h));
  e2h = iterateJacobi(e2h, r2h, 2*i, 1e-10);

  eh = interpolate(e2h);
  uh = uh + eh;

  averageError = mean(abs(uh - exactSolution));
  printf("Error after correction: %f\n", averageError);

  uh = iterateJacobi(uh, f, i, 1e-10);
  averageError = mean(abs(uh - exactSolution));
  printf("Error after post smoothing: %f\n", averageError);

  printf("\n");
endfor
