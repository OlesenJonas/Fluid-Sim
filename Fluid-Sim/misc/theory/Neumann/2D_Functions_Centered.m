% some assignment so that octave doesnt think this is a script file
x = 0;

%todo: all functions assume square domain, ie: dx == dy

function showScaled(X)
  X2 = X - min(min(X));
  X2 ./= max(max(X))-min(min(X));
  h = imshow(X2)
  waitfor(h)
endfunction

function x2 = negativeLaplace(x)
  x2 = x;
  h = 1.0/columns(x);

  imax = rows(x);
  jmax = columns(x);

  % much slower with all the ifs but simple to write, so idc

  for i = 1:rows(x)
    for j = 1:columns(x)

      left = j-1;
      if(left==0)
        left = 1;
      endif
      right = j+1;
      if(right==jmax+1)
        right = jmax;
      endif
      top = i-1;
      if(top==0)
        top = 1;
      endif
      bot = i+1;
      if(bot==imax+1)
        bot = imax;
      endif

      x2(i,j) = (-x(top,j) + 2*x(i,j) - x(bot,j))/(h*h) + (-x(i,right) + 2*x(i,j) - x(i,left))/(h*h);

    endfor
  endfor
endfunction

% jacobi iteration for -laplace matrix
function u2 = jacobi(u, f)
  u2 = u;
  h = 1.0/columns(u);

  imax = rows(u);
  jmax = columns(u);

  for i = 1:rows(u)
    for j = 1:columns(u)

      left = j-1;
      if(left==0)
        left = 1;
      endif
      right = j+1;
      if(right==jmax+1)
        right = jmax;
      endif
      top = i-1;
      if(top==0)
        top = 1;
      endif
      bot = i+1;
      if(bot==imax+1)
        bot = imax;
      endif

      u2(i,j) = f(i,j) - ( -1*u(top,j)/(h*h) + -1*u(bot,j)/(h*h) + -1*u(i,left)/(h*h) + -1*u(i,right)/(h*h));
      u2(i,j) = u2(i,j) / (4/(h*h));

    endfor
  endfor

endfunction

function u2h = restrict(uh)
  u2h = zeros(rows(uh)/2, columns(uh)/2);
  for i = 1:rows(u2h)
    for j = 1:columns(u2h)
      u2h(i,j) = uh(2*i, 2*j) + uh(2*i -1, 2*j) + uh(2*i, 2*j -1) + uh(2*i -1, 2*j -1);
      u2h(i,j) /= 4;
    endfor
  endfor
endfunction

function uh = interpolate(u2h)
  uh = zeros(rows(u2h)*2, columns(u2h)*2);

  % center interpolatation
  for i = 1:rows(uh)
    for j = 1:columns(uh)
        uh(i,j) = u2h(ceil(i/2), ceil(j/2));
    endfor
  endfor

endfunction

function u2 = iterateJacobi(u, f, steps, threshold)
  u2 = u;
  for i = 1:steps
    u2 = jacobi(u, f);
    f2 = negativeLaplace(u2);
    d = abs(f2 - f);
    if(max(max(d)) < threshold)
      break;
    endif
    u = u2;
  endfor
##  i
endfunction

function error = getErrorMinusConstantOffset(M1, M2)
  error = mean(mean(abs( (M1-min(min(M1))) - (M2-min(min(M2))) )));
endfunction
