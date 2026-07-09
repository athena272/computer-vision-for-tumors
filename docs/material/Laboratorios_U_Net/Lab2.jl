using LinearAlgebra,DelimitedFiles,PyPlot,Statistics
using Images, FileIO, NPZ

################## FUNÇÕES #####################
function leImagem(arquivo)
 x = load(arquivo);
 y = zeros(size(x,1),size(x,2),3);
 for i in 1:size(x,1)
  for j in 1:size(x,2)
   y[i,j,1:3] = [x[i,j].r; x[i,j].g; x[i,j].b];
  end
 end
return y
end

function geraApontadores(Mx,Nx,Kx,Mw,Nw,S)  
  Kw = Kx
  L = length(1:S:Mx)*length(1:S:Nx);
  pConv=zeros(Int64,L,Mw*Nw*Kw); # Cada linha representa uma posição da convolucao 
  aux = reshape(collect(1:Mx*Nx*Kx),Mx,Nx,Kx);
  margemM = Int(round((Mw-1)/2));
  margemN = Int(round((Nw-1)/2));
  cont = 1;
  for n in 1:S:Nx
    pN = collect(n-margemN:n+margemN);
	pp = findall(pN .<= 0); pN[pp] = -pN[pp].+1;
	pp = findall(pN .> Nx); pN[pp] = -pN[pp].+(2*Nx+1);	
	for m in 1:S:Mx # Avança na vertical primeiro
	  pM = collect(m-margemM:m+margemM);
	  pp = findall(pM .<= 0); pM[pp] = -pM[pp].+1;
	  pp = findall(pM .> Mx); pM[pp] = -pM[pp].+(2*Mx+1);
	  pConv[cont,:] = aux[pM,pN,:][:]';
	  cont += 1;
	end
  end
  return pConv
end

function geraApontadoresMaxpooling2x2(Mx,Nx,Kx)
  M = Mx*Nx*Kx;
  A = reshape(collect(1:M),Mx,Nx,Kx);
  My = Int(round(Mx/2)-1);
  Ny = Int(round(Nx/2)-1);
  Np = My*Ny;
  pX = zeros(Int,Np,4*Kx);
  cont = 1;
  for j in 1:2:Nx-1
    for i in 1:2:Mx-1
     pX[cont,:] .= A[i:i+1,j:j+1,:][:];
     cont = cont + 1;
    end
  end
  return pX
end

function maxpooling2x2(X,Mx,Nx,Kx,pX)
  My = Int(round(Mx/2)-1);
  Ny = Int(round(Nx/2)-1);
  Ky = Kx;
  Y = zeros(My*Ny,Kx);
  pMax = zeros(Int,size(pX,1),Kx);
  L = size(pX,1);
  k = 1;
  for j in 1:4:size(pX,2)-1
    pNaCamada = pX[:,j:j+3];
    val,pos = findmax(X[pNaCamada],dims=2);
    pMax[:,k] = pNaCamada[pos];
    Y[:,k] = val;
    k = k+1;
  end
  return Y,My,Ny,Ky,pMax
end

function geraApontadoresUpConv(Mx,Nx,Kx,My,Ny,Ky)
  M = My*Ny*Ky;
  A = reshape(collect(1:M),My,Ny,Ky);
  saltoM = My/Mx;
  saltoN = Ny/Nx;
  aM = Int.(collect(floor.(1:saltoM:My)));
  aN = Int.(collect(floor.(1:saltoN:Ny)));
  Np = length(aM)*length(aN);
  pY = zeros(Int,Np,9*Ky);
  cont = 1;
  for j in aN
    pN = collect((j-1):(j+1));
    pN[findall(pN .<= 0)] = -pN[findall(pN .<= 0)] .+ 1;
    pN[findall(pN .> Ny)] = -pN[findall(pN .> Ny)] .+ (2*Ny+1);
    for i in aM
     pM = collect((i-1):(i+1));
     pM[findall(pM .<= 0)] = -pM[findall(pM .<= 0)] .+ 1;
     pM[findall(pM .> My)] = -pM[findall(pM .> My)] .+ (2*My+1);
     pY[cont,:] = A[pM,pN,:][:];
     cont += 1;
    end
  end
  return pY
end

function upConv(X,Mx,Nx,Kx,My,Ny,Ky,pY,W)
  Y = zeros(My*Ny,Ky);
  aux = X*W';
  for k = 1:size(pY,1)
   Y[pY[k,:]] = Y[pY[k,:]]+aux[k,:];
  end
  return Y
end

function convVolta(Y,My,Ny,Ky,Kx,pX,W)
  X = zeros(My*Ny,Kx);
  acum = Y*W';
  for k = 1:size(pX,1)
   X[pX[k,:]] = X[pX[k,:]]+acum[k,:];
  end
  return X
end


################## PRINCIPAL #####################
	alfa = 1e-7;

	X = leImagem("copacabana.jpg");
	Mx,Nx,Kx = size(X);
	# Para ver a foto, use: imshow(X); show()

	# A será o alvo, e a entrada X é trocada por ruído uniforme entre 0 e 1:
	X[90:100,90:100,:].=0; #<- Falha artificial na imagem
	# A = copy(X); 
	# X = rand(Mx,Nx,Kx);

	# Usando representação puramente matricial:
	X = reshape(X,Mx*Nx,3);
	# A = reshape(A,Mx*Nx,3);

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
	Z1 = tanh.(X[pX]*W1.+B1);
	Mz1 = Mx; Nz1 = Nx; Kz1 = Nfiltros1;
	pZ1 = geraApontadoresMaxpooling2x2(Mz1,Nz1,Kz1);
	Z2,Mz2,Nz2,Kz2,pMax2 = maxpooling2x2(Z1,Mz1,Nz1,Kz1,pZ1);
	pZ2 = geraApontadores(Mz2,Nz2,Kz2,Mw,Nw,1);
	Z3 = tanh.(Z2[pZ2]*W2.+B2);
	Mz3 = Mz2; Nz3 = Nz2; Kz3 = Nfiltros2;
	pY = geraApontadoresUpConv(Mz3,Nz3,Kz3,Mx,Nx,Kx);
	Y = upConv(Z3,Mz3,Nz3,Kz3,Mx,Nx,Kx,pY,W3);
	#Y = tanh.(Y);

	## Volta:
	Ey = A-Y;
	#Ey = Ey.*(ones(size(Y)).- Y.^2); # "Pedágio" da tanh
	Ez3 = Ey[pY]*W3; # Volta do upConv
	Ez3 = Ez3.*(ones(size(Z3)).- Z3.^2); # "Pedágio" da tanh
	Ez2 = convVolta(Ez3,Mz3,Nz3,Kz3,Kz2,pZ2,W2); # Volta da conv.
	Ez1= zeros(size(Z1)); Ez1[pMax2] = Ez2; # Volta da maxpooling
	Ez1 = Ez1.*(ones(size(Z1)).-Z1.^2); # "Pedágio" da tanh
	#Ex = convVolta(Ez1,Mz1,Nz1,Kz1,Kx,pX,W1);

	##Ajuste da rede:
	W1 = W1 + alfa*(X[pX]'*Ez1);
	B1 = B1 + alfa*(ones(1,size(Ez1,1))*Ez1);
	W2 = W2 + alfa*(Z2[pZ2]'*Ez3);
	B2 = B2 + alfa*(ones(1,size(Ez3,1))*Ez3);
	W3 = W3 + alfa*Ey[pY]'*Z3;

function principal(X,Y,W1,B1,W2,B2,W3,B3,Nciclos,alfa)
	#Nciclos = 10;
	#alfa = 1e-6;
	J = zeros(Nciclos);
	figure(1)
	clf()
	for ciclo in 1:Nciclos
	  # Ida:
	  Z1 = tanh.(X[pX]*W1.+B1);
	  Z2,Mz2,Nz2,Kz2,pMax2 = maxpooling2x2(Z1,Mz1,Nz1,Kz1,pZ1);
	  Z3 = tanh.(Z2[pZ2]*W2.+B2);
	  pY = geraApontadoresUpConv(Mz3,Nz3,Kz3,Mx,Nx,Kx);
	  Y = upConv(Z3,Mz3,Nz3,Kz3,Mx,Nx,Kx,pY,W3);
	  #Y = tanh.(Y);

	  # Volta:
	  Ey = A-Y;
	  eqm = mean(Ey[:].^2);
	  J[ciclo] = eqm;
	  plot(ciclo,log10(J[ciclo]),marker = ".",markersize=0.5,color="b"); show()
	  println("Ciclo $ciclo, EQM = $eqm");
	  #Ey = Ey.*(ones(size(Y)).- Y.^2); # "Pedágio" da tanh
	  Ez3 = Ey[pY]*W3; # Volta do upConv
	  Ez3 = Ez3.*(ones(size(Z3)).- Z3.^2); # "Pedágio" da tanh
	  Ez2 = convVolta(Ez3,Mz3,Nz3,Kz3,Kz2,pZ2,W2); # Volta da conv.
	  Ez1= zeros(size(Z1)); Ez1[pMax2] = Ez2; # Volta da maxpooling
	  Ez1 = Ez1.*(ones(size(Z1)).-Z1.^2); # "Pedágio" da tanh
	  #Ex = convVolta(Ez1,Mz1,Nz1,Kz1,Kx,pX,W1);

	  #Ajuste da rede:
	  W1 = W1 + alfa*(X[pX]'*Ez1);
	  B1 = B1 + alfa*(ones(1,size(Ez1,1))*Ez1);
	  W2 = W2 + alfa*(Z2[pZ2]'*Ez3);
	  B2 = B2 + alfa*(ones(1,size(Ez3,1))*Ez3);
	  W3 = W3 + alfa*Ey[pY]'*Z3;
	end

	figure(2)
	clf()
	subplot(1,2,1),imshow(reshape(X,Mx,Nx,Kx))
	subplot(1,2,2),imshow(reshape(Y,Mx,Nx,Kx))
	show()
	return X,Y,W1,B1,W2,B2,W3,B3
end
X,Y,W1,B1,W2,B2,W3,B3 = principal(X,Y,W1,B1,W2,B2,W3,B3,20,3e-6);


