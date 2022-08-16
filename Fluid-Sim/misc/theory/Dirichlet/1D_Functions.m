% some assignment so that octave doesnt think this is a script file
x = 0;

function showScaled(X)
  X2 = X - min(min(X));
  X2 ./= max(max(X))-min(min(X));
  h = imshow(X2)
  waitfor(h)
endfunction

function x2 = negativeLaplaceWithoutBorder(x)
  x2 = x;
  h = 1.0/(columns(x)-1);
  for i = 2:columns(x)-1
    x2(i) = (-x(i+1) + 2*x(i) - x(i-1))/(h*h);
  endfor
endfunction

% jacobi iteration for -laplace matrix
function u2 = jacobiWithoutBorder(u, f)
  u2 = u;
  h = 1.0/(columns(u)-1);
  for i = 2:columns(u)-1
    u2(i) = f(i) - ( -1*u(i-1)/(h*h) + -1*u(i+1)/(h*h));
    u2(i) = u2(i) / (2/(h*h));
  endfor
endfunction

function u2h = restrict(uh)
  u2h = zeros(1, (columns(uh)-1)/2 +1);
  u2h(1) = uh(1);
  u2h(columns(u2h)) = uh(columns(uh));
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
    u2 = jacobiWithoutBorder(u, f);
    f2 = negativeLaplaceWithoutBorder(u2);
    d = abs(f2(2:columns(f)-1) - f(2:columns(f)-1));
    if(max(max(d)) < threshold)
      break;
    endif
    u = u2;
  endfor
##  i
endfunction
