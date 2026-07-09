function pY = geraApontadoresUpConv(Mx,Nx,Kx,My,Ny,Ky)
  M = My*Ny*Ky;
  A = reshape(1:M,My,Ny,Ky);
  saltoM = My/Mx;
  saltoN = Ny/Nx;
  aM = floor(1:saltoM:My);
  aN = floor(1:saltoN:Ny);
  Np = length(aM)*length(aN);
  pY = zeros(Np,9*Ky);
  cont = 1;
  for j = aN
    pN = (j-1):(j+1);
    pN(pN<=0)= -pN(pN<=0)+1;
    pN(pN>Ny)= -pN(pN>Ny)+2*Ny+1;
    for i = aM
     pM = (i-1):(i+1);
     pM(pM<=0)= -pM(pM<=0)+1;
     pM(pM>My)= -pM(pM>My)+2*My+1;
     pY(cont,:) = A(pM,pN,:)(:)';
     cont = cont + 1;
    endfor
  endfor
endfunction

