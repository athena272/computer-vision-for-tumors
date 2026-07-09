function [pX] = geraApontadoresMaxpooling2x2(Mx,Nx,Kx)
  M = Mx*Nx*Kx;
  A = reshape(1:M,Mx,Nx,Kx);
  My = (round(Mx/2)-1);
  Ny = (round(Nx/2)-1);
  Np = My*Ny;
  pX = zeros(Np,4*Kx);
  cont = 1;
  for j = 1:2:Nx-1
    for i = 1:2:Mx-1
     pX(cont,:) = A(i:i+1,j:j+1,:)(:)';
     cont = cont + 1;
    endfor
  endfor
endfunction

