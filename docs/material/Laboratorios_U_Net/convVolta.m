function [X] = convVolta(Y,My,Ny,Ky,Kx,pX,W)
  X = zeros(My*Ny,Kx);
  acum = Y*W';
  for k = 1:size(pX,1)
   X(pX(k,:)) = X(pX(k,:))+acum(k,:);
  end
endfunction


