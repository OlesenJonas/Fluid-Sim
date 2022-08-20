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
  h = 1.0/(columns(x)-1);

  imax = rows(x);
  jmax = columns(x);

  % much slower with all the ifs but simple to write, so idc

  for i = 1:rows(x)
    for j = 1:columns(x)

      left = j-1;
      if(j==1)
        left = 2;
      endif
      right = j+1;
      if(j==jmax)
        right = jmax-1;
      endif
      top = i-1;
      if(i==1)
        top = 2;
      endif
      bot = i+1;
      if(i==imax)
        bot = imax-1;
      endif

      x2(i,j) = (-x(top,j) + 2*x(i,j) - x(bot,j))/(h*h) + (-x(i,right) + 2*x(i,j) - x(i,left))/(h*h);

    endfor
  endfor
endfunction

% jacobi iteration for -laplace matrix
function u2 = jacobi(u, f)
  u2 = u;
  h = 1.0/(columns(u)-1);

  imax = rows(u);
  jmax = columns(u);

  for i = 1:rows(u)
    for j = 1:columns(u)

      left = j-1;
      if(j==1)
        left = 2;
      endif
      right = j+1;
      if(j==jmax)
        right = jmax-1;
      endif
      top = i-1;
      if(i==1)
        top = 2;
      endif
      bot = i+1;
      if(i==imax)
        bot = imax-1;
      endif

      u2(i,j) = f(i,j) - ( -1*u(top,j)/(h*h) + -1*u(bot,j)/(h*h) + -1*u(i,left)/(h*h) + -1*u(i,right)/(h*h));
      u2(i,j) = u2(i,j) / (4/(h*h));

    endfor
  endfor

endfunction

function u2h = restrict(uh)
  u2h = zeros((rows(uh)-1)/2 +1, (columns(uh)-1)/2 +1);

  imax = rows(uh);
  jmax = columns(uh);

  % interpolate values inside
  for i = 1:rows(u2h)
    for j = 1:columns(u2h)
      top = (i-1)*2;
      centerv = top+1;
      bot = top+2;
      left = (j-1)*2;
      centerh = left+1;
      right = left+2;

      if(left==0)
        left = 2;
      endif
      if(right==jmax+1)
        right = jmax-1;
      endif
      if(top==0)
        top = 2;
      endif
      if(bot==imax+1)
        bot = imax-1;
      endif

      u2h(i,j) = uh(top, left) + uh(bot, left) + uh(top, right) + uh(bot, right);
      u2h(i,j) += 2*uh(top, centerh) + 2*uh(bot, centerh) + 2*uh(centerv, left) + 2*uh(centerv, right);
      u2h(i,j) += 4*uh(centerv, centerh);
      u2h(i,j) /= 16;
    endfor
  endfor
endfunction

function uh = interpolate(u2h)
  uh = zeros((rows(u2h)-1)*2 +1, (columns(u2h)-1)*2 +1);

  % exact matches
  for i = 1:2:rows(uh)
    for j = 1:2:columns(uh)
      uh(i,j) = u2h((i+1)/2,(j+1)/2);
    endfor
  endfor

  % horizontal interpolatation
  for i = 1:2:rows(uh)
    for j = 2:2:columns(uh)
      uh(i,j) = 0.5*u2h((i+1)/2,j/2) + 0.5*u2h((i+1)/2,j/2 + 1);
    endfor
  endfor

  % horizontal interpolatation
  for i = 2:2:rows(uh)
    for j = 1:2:columns(uh)
      uh(i,j) = 0.5*u2h(i/2,(j+1)/2) + 0.5*u2h(i/2 + 1,(j+1)/2);
    endfor
  endfor

  % center interpolatation
  for i = 2:2:rows(uh)
    for j = 2:2:columns(uh)
      left = j/2;
      right = left + 1;
      top = i/2;
      bot = top + 1;
      uh(i,j) = 0.25*u2h(top, left) + 0.25*u2h(top, right) + 0.25*u2h(bot, left) + 0.25*u2h(bot, right);
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
