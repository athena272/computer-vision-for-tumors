function [Y] = upConv(X,Mx,Nx,Kx,My,Ny,Ky,pY,W)
  Y = zeros(My*Ny,Ky);
##  aux = zeros(size(pY));
##  for canal = 1:size(W,2)
##   aux = aux+X(:,canal)*W(:,canal)';
##  end
  aux = X*W';
  for k = 1:size(pY,1)
   Y(pY(k,:)) = Y(pY(k,:))+aux(k,:);
  end
endfunction


