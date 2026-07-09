function pX = geraApontadores(LinhasX,ColunasX,CamadasX,LinhasW,ColunasW,Salto)
  M = LinhasX*ColunasX*CamadasX;
  A = reshape(1:M,LinhasX,ColunasX,CamadasX);
  cont = 1;
  clear pX
  for j = 1:Salto:ColunasX-ColunasW+1
    for i = 1:Salto:LinhasX-LinhasW+1
     pX(cont,:) = A(i:i+LinhasW-1,j:j+ColunasW-1,:)(:)';
     cont = cont + 1;
    endfor
  endfor
endfunction

