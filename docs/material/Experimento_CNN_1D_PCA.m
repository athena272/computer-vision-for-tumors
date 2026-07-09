x = load('VO00001T001_BioChaves.txt');
plot(x)
soundsc(x,22050)

# Primeira camada de projeção:
Jan = 500;
P = [];
for n = 1:50:length(x)-Jan+1
 P = [P; x(n:n+Jan-1)'];
end
P = P-mean(P,1);
C = (1/size(P,1))*P'*P;
[vet,val] = eig(C);
val = real(val);
vet = real(vet);

#plot(diag(val))
K1 = 100;
W1 = vet(:,1:K1);

z1 = [];
for n = 1:Jan:length(x)-Jan+1
 z1=[z1 W1'*x(n:n+Jan-1)];
end
z1 = max(0,z1);

##plot(x(1:Jan))
##hold on
##plot(W1*z1(:,1))
##
##clf
##plot(x(1001:1500))
##hold on
##plot(W1*z1(:,3))
##
##imagesc(z1), colormap(gray)

# Segunda camada de projeção:
P = [];
for n = 1:size(z1,2)-2
 P = [P; z1(:,n:n+2)(:)'];
end
P = P-mean(P,1);
C = (1/size(P,1))*P'*P;
[vet,val] = eig(C);
val = real(val);
vet = real(vet);

K2 = 90;
W2 = vet(:,1:K2);
z2 = [];
for n = 1:3:size(z1,2)-2
 z2 = [z2 W2'*z1(:,n:n+2)(:)];
end
##imagesc(z2), colormap(gray)
z2 = max(0,z2);

### Decodificação:
##z1r = W2*z2(:,1);
##z1r = reshape(z1r,K1,3);
##xr = W1*z1r;
##xr = reshape(xr,1,3*Jan);
##plot(xr)
##hold on
##plot(x(1:3*Jan))


# Terceira camada de projeção:
##size(z2)
##ans =
##   70   44
P = [];
for n = 1:size(z2,2)-2
 P = [P; z2(:,n:n+2)(:)'];
end
P = P-mean(P,1);
C = (1/size(P,1))*P'*P;
[vet,val] = eig(C);
val = real(val);
vet = real(vet);

##sum(diag(val)(1:35))/sum(diag(val))
##ans = 0.9984
K3 = 35;
W3 = vet(:,1:K3);
z3 = [];
for n = 1:3:size(z2,2)-2
 z3 = [z3 W3'*z2(:,n:n+2)(:)];
end
##imagesc(z3), colormap(gray)

### Decodificação parcial:
##z2r = W3*z3(:,1);
##z2r = reshape(z2r,K2,3);
##z1r = W2*z2r;
##z1r = reshape(z1r,K1,3*3);
##xr = W1*z1r;
##xr = reshape(xr,1,3*3*Jan);
##plot(xr)
##hold on
##plot(x(1:3*3*Jan))


### Filtragem profunda:
##Lim = 0.5*std(z3);
##z3(abs(z3)<Lim)=0;

##z3(1:2,:) = 0;
##z3(6:end,:) = 0;

z3 = max(0,z3);

### Decodificação completa:
xr = zeros(length(x),1);
for k = 1:size(z3,2)
 z2r = W3*z3(:,k);
 z2r = reshape(z2r,K2,3);
 z1r = W2*z2r;
 z1r = reshape(z1r,K1,3*3);
 aux = W1*z1r;
 xr((k-1)*(3*3*Jan)+1:k*(3*3*Jan)) = reshape(aux,1,(3*3*Jan));
end
soundsc(xr,22050)
hold on
plot(xr)

