# Laboratório #0 de autocodificação
clear
# Leitura de apenas uma imagem
X = imread('copacabana.jpg');
[Mx,Nx,Kx] = size(X);

#Z = randn(size(X));

# Para fins de processamento da imagem, X deve ser convertido em double e normalizada:
X = (1/255)*double(X);
# Obs.: use image(uint8(255*X)) para converter X de volta a 8 bits e visualizá-la
X = reshape(X,Mx*Nx,3);

Nfiltros = 3; # Escolher aqui o número de filtros de convolução
# Dimensões dos filtros:
Mw = 3;
Nw = 3;

# Inicializando os filtros aleatoriamente:
W = (1/sqrt(Mw*Nw*Kx+1))*randn(Mw*Nw*Kx,Nfiltros);
B = (1/sqrt(Mw*Nw*Kx+1))*randn(1,Nfiltros);

# Inicializando os apontadores:
pX = geraApontadores(Mx,Nx,Kx,Mw,Nw,1);

figure(1)
for laco = 1:300
  # Convolução:
  Y = tanh(X(pX)*W+B);

  # Volta do erro:
  Ey = X - Y;
  J(laco) = mean(Ey(:).^2);
  #disp(J(laco));
  plot(J);
  drawnow
  Ey = Ey.*(1-Y.^2);
  alfa = 1e-6;
  W = W + alfa*(X(pX)'*Ey);
  B = B + alfa*(ones(1,size(Ey,1))*Ey);
end

figure(2)
subplot(1,2,1),imagesc(reshape(X,Mx,Nx,Kx))
subplot(1,2,2),imagesc(reshape(Y,Mx,Nx,Kx))

