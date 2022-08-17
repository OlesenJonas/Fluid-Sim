% some assignment so that octave doesnt think this is a script file
x = 0;

function showScaled(X)
  X2 = X - min(min(X));
  X2 ./= max(max(X))-min(min(X));
  h = imshow(X2)
  waitfor(h)
endfunction

function x2 = negativeLaplace(x)
  x2 = zeros(rows(x), columns(x));
  h = 1.0/(columns(x)-1);

  % endpoints
  x2(1) = ( 2*x(1) - 2*x(2) )/(h*h);
  x2(columns(x2)) = ( 2*x(columns(x)) - 2*x(columns(x)-1) )/(h*h);

  % inner
  for i = 2:columns(x)-1
    x2(i) = (-x(i+1) + 2*x(i) - x(i-1))/(h*h);
  endfor

endfunction

% jacobi iteration for -laplace matrix
function u2 = jacobi(u, f)
  u2 = zeros(rows(u), columns(u));
  h = 1.0/(columns(u)-1);

  % endpoints
  u2(1) = f(1) - ( -2*u(2)/(h*h));
  u2(1) = u2(1) / (2/(h*h));
  u2(columns(u2)) = f(columns(u2)) - ( -2*u(columns(u2)-1)/(h*h));
  u2(columns(u2)) = u2(columns(u2)) / (2/(h*h));

  % inner
  for i = 2:columns(u)-1
    u2(i) = f(i) - ( -1*u(i-1)/(h*h) + -1*u(i+1)/(h*h));
    u2(i) = u2(i) / (2/(h*h));
  endfor
endfunction

function u2h = restrict(uh)
  u2h = zeros(1, (columns(uh)-1)/2 +1);

  u2h(1) = uh(1);
  u2h(columns(u2h)) = uh(columns(uh));
  % this is more accurate to the formulas I think since it takes the "ghost points" into account
  % but quick test here is actually worse, not sure how it holds up in 2D
##  u2h(1) = 0.5*uh(1) + 0.5*uh(2);
##  u2h(columns(u2h)) = 0.5*uh(columns(uh)) + 0.5*uh(columns(uh)-1);

  for i = 2:columns(u2h)-1
    u2h(i) = 0.25 * uh( (i-1)*2 ) + 0.5 * uh( (i-1)*2 + 1) + 0.25 * uh( (i-1)*2 + 2 );
  endfor

endfunction

function uh = interpolate(u2h)
  uh = zeros(1, (columns(u2h)-1)*2 +1);

  for i = 2:2:columns(uh)
    uh(i) = 0.5 * u2h(i/2) + 0.5 * u2h(i/2 + 1);
  endfor

  for i = 1:2:columns(uh)
    uh(i) = u2h( (i+1)/2 );
  endfor

endfunction

function u2 = iterateJacobi(u, f, steps, threshold)
  u2 = u;
  for i = 1:steps
    u2 = jacobi(u, f);
    f2 = negativeLaplace(u2);
    d = abs(f2 - f);
##    d
    if(max(max(d)) < threshold)
      break;
    endif
    u = u2;
  endfor
##  i
endfunction

function error = getErrorMinusConstantOffset(M1, M2)
  error = mean(abs( (M1-min(min(M1))) - (M2-min(min(M2))) ));
endfunction
