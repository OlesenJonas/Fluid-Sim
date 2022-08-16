% some assignment so that octave doesnt think this is a script file
x = 0;

%todo: all functions assume square domain, ie: dx == dy

function showScaled(X)
  X2 = X - min(min(X));
  X2 ./= max(max(X))-min(min(X));
  h = imshow(X2)
  waitfor(h)
endfunction

function x2 = negativeLaplaceWithoutBorder(x)
  x2 = x;
  h = 1.0/(columns(x)-1);
  for i = 2:rows(x)-1
    for j = 2:columns(x)-1
      x2(i,j) = (-x(i+1,j) + 2*x(i,j) - x(i-1,j))/(h*h) + (-x(i,j+1) + 2*x(i,j) - x(i,j-1))/(h*h);
    endfor
  endfor
endfunction

% jacobi iteration for -laplace matrix
function u2 = jacobiWithoutBorder(u, f)
  u2 = u;
  h = 1.0/(columns(u)-1);
  for i = 2:rows(u)-1
    for j = 2:columns(u)-1
      u2(i,j) = f(i,j) - ( -1*u(i-1,j)/(h*h) + -1*u(i+1,j)/(h*h) + -1*u(i,j-1)/(h*h) + -1*u(i,j+1)/(h*h));
      u2(i,j) = u2(i,j) / (4/(h*h));
    endfor
  endfor
endfunction

function u2h = restrict(uh)
  u2h = zeros((rows(uh)-1)/2 +1, (columns(uh)-1)/2 +1);

  % corners stay the same
  u2h(1,1) = uh(1,1);
  u2h(1,columns(u2h)) = uh(1,columns(uh));
  u2h(rows(u2h),1) = uh(rows(uh),1);
  u2h(rows(u2h),columns(u2h)) = uh(rows(uh),columns(uh));

  % interpolate values on horizontal borders
  max2i = rows(u2h);
  maxi = rows(uh);
  for j = 2:columns(u2h)-1
    u2h(1,j) = 0.25 * uh(1, (j-1)*2 ) + 0.5 * uh(1, (j-1)*2 + 1) + 0.25 * uh(1, (j-1)*2 + 2 );
    u2h(max2i,j) = 0.25 * uh(maxi, (j-1)*2 ) + 0.5 * uh(maxi, (j-1)*2 + 1) + 0.25 * uh(maxi, (j-1)*2 + 2 );
  endfor

  % interpolate values on vertical borders
  max2j = columns(u2h);
  maxj = columns(uh);
  for i = 2:rows(u2h)-1
    u2h(i,1) = 0.25 * uh((i-1)*2, 1) + 0.5 * uh((i-1)*2 + 1, 1) + 0.25 * uh((i-1)*2 + 2, 1);
    u2h(i,max2j) = 0.25 * uh((i-1)*2, maxj) + 0.5 * uh((i-1)*2 + 1, maxj) + 0.25 * uh((i-1)*2 + 2, maxj);
  endfor

  % interpolate values inside
  for i = 2:rows(u2h)-1
    for j = 2:columns(u2h)-1
      top = (i-1)*2;
      left = (j-1)*2;
      u2h(i,j) = uh(top, left) + uh(top+2, left) + uh(top, left+2) + uh(top+2, left+2);
      u2h(i,j) += 2*uh(top, left+1) + 2*uh(top+1, left) + 2*uh(top+1, left+2) + 2*uh(top+2, left+1);
      u2h(i,j) += 4*uh(top+1, left+1);
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
    u2 = jacobiWithoutBorder(u, f);
    f2 = negativeLaplaceWithoutBorder(u2);
    d = abs(f2(2:rows(f)-1, 2:columns(f)-1) - f(2:rows(f)-1, 2:columns(f)-1));
    if(max(max(d)) < threshold)
      break;
    endif
    u = u2;
  endfor
##  i
endfunction
