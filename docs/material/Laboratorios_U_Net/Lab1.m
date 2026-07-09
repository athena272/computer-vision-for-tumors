# Laboratório #0 de autocodificação
clear
# Leitura de apenas uma imagem
X = imread('copacabana.jpg');
[Mx,Nx,Kx] = size(X);

# Para fins de processamento da imagem, X deve ser convertido em double e normalizada:
X = (1/255)*double(X);
# Obs.: use image(uint8(255*X)) para converter X de volta a 8 bits e visualizá-la
X(90:100,90:100,:)=0; #<- Falha artificial na imagem
X = reshape(X,Mx*Nx,3);


Nfiltros1 = 10; # Escolher aqui o número de filtros de convolução
Nfiltros2 = 3;

# Dimensões dos filtros:
Mw = 11;
Nw = 11;

# Inicializando os filtros aleatoriamente:
W1 = (1/sqrt(Mw*Nw*Kx+1))*randn(Mw*Nw*Kx,Nfiltros1);
B1 = (1/sqrt(Mw*Nw*Kx+1))*randn(1,Nfiltros1);
W2 = (1/sqrt(Mw*Nw*Nfiltros1+1))*randn(Mw*Nw*Nfiltros1,Nfiltros2);
B2 = (1/sqrt(Mw*Nw*Nfiltros1+1))*randn(1,Nfiltros2);

# Inicializando os apontadores:
pX = geraApontadores(Mx,Nx,Kx,Mw,Nw,1);
Mz = Mx; Nz = Nx; Kz = Nfiltros1;
pZ = geraApontadores(Mz,Nz,Kz,Mw,Nw,1);

alfa = 1e-7;
figure(1)
for laco = 1:50
  # Convolução:
  Z = tanh(X(pX)*W1+B1);
  Y = tanh(Z(pZ)*W2+B2);

  # Volta do erro:
  Ey = X - Y;
  J(laco) = mean(Ey(:).^2);
  #disp(J(laco));
  plot(J);
  drawnow
  Ey = Ey.*(1-Y.^2);
  [Ez] = convVolta(Ey,Mx,Nx,Kx,Kz,pZ,W2);
  W1 = W1 + alfa*(X(pX)'*Ez);
  B1 = B1 + alfa*(ones(1,size(Ez,1))*Ez);
  W2 = W2 + alfa*(Z(pZ)'*Ey);
  B2 = B2 + alfa*(ones(1,size(Ey,1))*Ey);
end

figure(2)
subplot(1,2,1),imagesc(reshape(X,Mx,Nx,Kx))
subplot(1,2,2),imagesc(reshape(Y,Mx,Nx,Kx))

