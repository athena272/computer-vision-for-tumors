function [Y,My,Ny,Ky,pMax] = maxpooling2x2(X,Mx,Nx,Kx,pX)
  My = round(Mx/2)-1;
  Ny = round(Nx/2)-1;
  Ky = Kx;
  Y = zeros(My*Ny,Kx);
  pMax = zeros(size(pX,1),Kx);
  L = size(pX,1);
  k = 1;
  for j = 1:4:size(pX,2)-1
    pNaCamada = pX(:,j:j+3);
    [val,pos] = max(X(pNaCamada)');
    pMax(:,k) = pNaCamada((pos-1)*L+(1:L));
    Y(:,k) = val;
    k = k+1;
  endfor
end



