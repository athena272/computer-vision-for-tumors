# Laboratório #0 de autocodificação
clear
# Leitura de apenas uma imagem
X = imread('copacabana.jpg');
[Mx,Nx,Kx] = size(X);
alfa = 3e-6;

# Para fins de processamento da imagem, X deve ser convertido em double e normalizada:
X = (1/255)*double(X);
# Obs.: use image(uint8(255*X)) para converter X de volta a 8 bits e visualizá-la
X(90:100,90:100,:)=0; #<- Falha artificial na imagem

A = X;
X = rand(size(X));

X = reshape(X,Mx*Nx,3);
A = reshape(A,Mx*Nx,3);

Nfiltros1 = 30; # Escolher aqui o número de filtros de convolução
Nfiltros2 = 60;
Nfiltros3 = 3;

# Dimensões dos filtros:
Mw = 3;
Nw = 3;

# Inicializando os filtros aleatoriamente:
W1 = (1/sqrt(Mw*Nw*Kx+1))*randn(Mw*Nw*Kx,Nfiltros1);
B1 = (1/sqrt(Mw*Nw*Kx+1))*randn(1,Nfiltros1);
W2 = (1/sqrt(Mw*Nw*Nfiltros1+1))*randn(Mw*Nw*Nfiltros1,Nfiltros2);
B2 = (1/sqrt(Mw*Nw*Nfiltros1+1))*randn(1,Nfiltros2);
W3 = (1/sqrt(Mw*Nw*Nfiltros2+1))*randn(Mw*Nw*Nfiltros3,Nfiltros2);
B3 = (1/sqrt(Mw*Nw*Nfiltros2+1))*randn(1,Nfiltros2);

# Primeiro ciclo e inicialização dos apontadores:
# Ida:
pX = geraApontadores(Mx,Nx,Kx,Mw,Nw,1);
Z1 = tanh(X(pX)*W1+B1);
Mz1 = Mx; Nz1 = Nx; Kz1 = Nfiltros1;
[pZ1] = geraApontadoresMaxpooling2x2(Mz1,Nz1,Kz1);
[Z2,Mz2,Nz2,Kz2,pMax2] = maxpooling2x2(Z1,Mz1,Nz1,Kz1,pZ1);
pZ2 = geraApontadores(Mz2,Nz2,Kz2,Mw,Nw,1);
Z3 = tanh(Z2(pZ2)*W2+B2);
Mz3 = Mz2; Nz3 = Nz2; Kz3 = Nfiltros2;
pY = geraApontadoresUpConv(Mz3,Nz3,Kz3,Mx,Nx,Kx);
[Y] = upConv(Z3,Mz3,Nz3,Kz3,Mx,Nx,Kx,pY,W3);
#Y = tanh(Y);

# Volta:
Ey = A-Y;
#Ey = Ey.*(1-Y.^2); # "Pedágio" da tanh
Ez3 = Ey(pY)*W3; # Volta do upConv
Ez3 = Ez3.*(1-Z3.^2); # "Pedágio" da tanh
[Ez2] = convVolta(Ez3,Mz3,Nz3,Kz3,Kz2,pZ2,W2); # Volta da conv.
Ez1= zeros(size(Z1)); Ez1(pMax2) = Ez2; # Volta da maxpooling
Ez1 = Ez1.*(1-Z1.^2); # "Pedágio" da tanh
#[Ex] = convVolta(Ez1,Mz1,Nz1,Kz1,Kx,pX,W1);
#Ajuste da rede:
W1 = W1 + alfa*(X(pX)'*Ez1);
B1 = B1 + alfa*(ones(1,size(Ez1,1))*Ez1);
W2 = W2 + alfa*(Z2(pZ2)'*Ez3);
B2 = B2 + alfa*(ones(1,size(Ez3,1))*Ez3);
W3 = W3 + alfa*Ey(pY)'*Z3;

Nciclos = 10000;
#J = zeros(Nciclos);
figure(1)
for laco = 1:Nciclos
  # Ida:
  Z1 = tanh(X(pX)*W1+B1); #Primeira convolução
  [Z2,Mz2,Nz2,Kz2,pMax2] = maxpooling2x2(Z1,Mz1,Nz1,Kz1,pZ1);
  Z3 = tanh(Z2(pZ2)*W2+B2); #Segunda convolução
  [Y] = upConv(Z3,Mz3,Nz3,Kz3,Mx,Nx,Kx,pY,W3); #convolução transposta
#  Y = tanh(Y);

  # Volta:
  Ey = A-Y;
  J(laco) = mean(Ey(:).^2);
  #disp(J(laco));
  plot(log(J));
  grid
  drawnow
#  Ey = Ey.*(1-Y.^2); # "Pedágio" da tanh
  Ez3 = Ey(pY)*W3; # Volta da convolução transposta
  Ez3 = Ez3.*(1-Z3.^2); # "Pedágio" da tanh
  [Ez2] = convVolta(Ez3,Mz3,Nz3,Kz3,Kz2,pZ2,W2); # Volta da conv.
  Ez1= zeros(size(Z1)); Ez1(pMax2) = Ez2; # Volta da maxpooling
  Ez1 = Ez1.*(1-Z1.^2); # "Pedágio" da tanh
  #[Ex] = convVolta(Ez1,Mz1,Nz1,Kz1,Kx,pX,W1);
  #Ajuste da rede:
  W1 = W1 + alfa*(X(pX)'*Ez1);
  B1 = B1 + alfa*(ones(1,size(Ez1,1))*Ez1);
  W2 = W2 + alfa*(Z2(pZ2)'*Ez3);
  B2 = B2 + alfa*(ones(1,size(Ez3,1))*Ez3);
  W3 = W3 + alfa*Ey(pY)'*Z3;
end

figure(2)
subplot(1,2,1),imagesc(reshape(X,Mx,Nx,Kx))
subplot(1,2,2),imagesc(reshape(Y,Mx,Nx,Kx))

