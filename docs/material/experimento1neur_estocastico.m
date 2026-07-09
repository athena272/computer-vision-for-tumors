##function y=f(x,w,b)
## aux = w*x+b;
## y = 1/(1+exp(-2*aux));
##end

x =[0.8; 1.1; 0.4; -0.2]
c =[1;1;0;0];
y = zeros(4,1);
e = zeros(4,1);
w = randn(1,1);
b = randn(1,1);
passo = 0.9; # O passo é grande porque o problema é fácil de otimizar
NumeroDeBateladas = 1000;
J = zeros(NumeroDeBateladas);
disp('Aprendendo...')
for batelada = 1:NumeroDeBateladas
 deltaw = 0;
 deltab = 0;
 [~,sorteio] = sort(rand(4,1)); # Sorteio aleatório da ordem dos padrões
 for k = 1:4
  n = sorteio(k);
  y(n) = f(x(n),w,b);
  e(n) = c(n)-y(n);
  # Versão estocástica do aprendizado por gradiente:
  #  w e b são ajustados para cada escolha aleatória de um padrão de treino
  w = w + e(n)*(1-y(n))*y(n)*x(n);
  b = b + e(n)*(1-y(n))*y(n);
 end
 J(batelada) = mean(e.^2);
end
plot(J)
xlabel('bateladas')
ylabel('Erro quadrático médio (J)')
disp('desejado obtido');
disp([c y])


