function pX = geraApontadoresCentrados(LinhasX,ColunasX,CamadasX,LinhasW,ColunasW,Salto)
  M = LinhasX*ColunasX*CamadasX;
  A = reshape(1:M,LinhasX,ColunasX,CamadasX);
  cont = 1;
  clear pX
  for j = 1:Salto:ColunasX
    colunas = j-round((ColunasW-1)/2):j+round((ColunasW-1)/2);
    pborda = find(colunas <= 0);
    colunas(pborda) = -colunas(pborda)+1;
    pborda = find(colunas > ColunasX);
    colunas(pborda) = 2*ColunasX -1 - colunas(pborda);
    for i = 1:Salto:LinhasX
     linhas = i-round((LinhasW-1)/2):i+round((LinhasW-1)/2);
     pborda = find(linhas<=0);
     linhas(pborda) = abs(linhas(pborda))+1;
     pborda = find(linhas > LinhasX);
     linhas(pborda) = 2*LinhasX - linhas(pborda);
     pX(cont,:) = A(linhas,colunas,:)(:)';
     cont = cont + 1;
    endfor
  endfor
endfunction

