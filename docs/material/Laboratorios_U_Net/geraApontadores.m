function pX = geraApontadores(Mx,Nx,Kx,Mw,Nw,S)
  M = Mx*Nx*Kx;
  A = reshape(1:M,Mx,Nx,Kx);
  Np = length(1:S:Mx)*length(1:S:Nx);
  pX = zeros(Np,Mw*Nw*Kx);
  margemM = floor((Mw-1)/2);
  margemN = floor((Nw-1)/2);
  cont = 1;
  for j = 1:S:Nx
    pN = (j-margemN):(j+margemN);
    pN(pN<=0)= -pN(pN<=0)+1;
    pN(pN>Nx)= -pN(pN>Nx)+2*Nx+1;
    for i = 1:S:Mx
     pM = (i-margemM):(i+margemM);
     pM(pM<=0)= -pM(pM<=0)+1;
     pM(pM>Mx)= -pM(pM>Mx)+2*Mx+1;
     pX(cont,:) = A(pM,pN,:)(:)';
     cont = cont + 1;
    endfor
  endfor
endfunction

