# Estudo do problema XOR:
# Definição de entradas e saídas desejadas:
X = [0 0;
     1 0;
     0 1;
     1 1];

O = [0
     1
     1
     0];

# Definição de função degrau:
function y = degrau(x)
   y = (x >= 0);
endfunction

# Solução analítica do problema XOR
W1 = [1 1;
      1 1];
B1 = [-1/2;
      -3/2];
W2 = [1 -1];
B2 = [-1/2];

# Experimente esta solução analítica assim:
## Z = degrau(W1*X'+B1)';
## Y = degrau(W2*Z'+B2)';
# e compare Y com as saídas desejadas em O.
# O erro é nulo, graças à camada escondida que projeta X em Z, "dobrando"
# o espaço de representação de X, onde o problema de classificação
# não é linearmente separável, em um novo espaço de representação de Z,
# onde as classes se tornam linearmente separáveis.


# Agora vamos à solução via aprendizado de máquina (com gradiente estocástico):
# Execute o código abaixo algumas vezes, e observe como a convergência
# ao erro quadrático nulo pode ser difícil, mesmo para um problema tão simples,
# para o qual pelo menos uma solução analítica foi encontrada facilmente (acima).

# Definição de função logística (que substitui o degrau por causa da derivada):
function y = logi(x)
   y = (1+exp(-x)).^(-1);
endfunction

# Inicialização aleatória dos pesos (parâmetros da rede):
W1 = randn(2,2);
B1 = randn(2,1);
W2 = randn(1,2);
B2 = randn(1,1);

passo = 50.0; #Passo de aprendizado
NumeroCiclos = 5000;
J = zeros(1,NumeroCiclos);
Ac = zeros(1,NumeroCiclos);
for ciclo = 1:NumeroCiclos
  ap = randperm(length(O)); # Apontador aleatório
  for k = ap
    # Fluxo de ida:
    x = X(k,:)';
    z = logi(W1*x+B1);
    y = logi(W2*z+B2);

    # Fluxo de volta do erro:
    e = O(k)-y;
    J(ciclo) = J(ciclo)+e^2;
    Ac(ciclo) =  Ac(ciclo) + (O(k) == (y >= 0.5));
    e2 = e*(1-y)*y; # erro retropropagado até a camada 2
    e1 = e2*W2'.*((1-z).*z); # erro retropropagado até a camada 1

    # Ajuste dos parâmetros:
    B2 = B2 + passo*e2;
    W2 = W2 + passo*e2*z';
    B1 = B1 + passo*e1;
    W1 = W1 + passo*e1*x';
  end
  J(ciclo) = J(ciclo)/length(O); #Erro quadrático médio do ciclo
  Ac(ciclo) = Ac(ciclo)/length(O); #Acurácia média do ciclo
end
subplot(2,1,1)
plot(J)
title("Acompanhamento do treinamento pelo EQM");
xlabel("Ciclos ou epochs de aprendizado")
ylabel("Erro quadrático médio (EQM) do ciclo")
subplot(2,1,2)
plot(Ac)
title("Acompanhamento do treinamento pela Acurácia");
xlabel("Ciclos ou epochs de aprendizado")
ylabel("Acurácia média do ciclo")

