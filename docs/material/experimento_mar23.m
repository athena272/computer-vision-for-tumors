# Normalização dos dados:
X = [X0; X1];
m = mean(X);
s = std(X);
X = X - m;
X = X/s;
w = 0.01*randn;
b = 0.01*randn;

D = [zeros(size(X0)); ones(size(X1))];
alfa = 0.01;
Nciclos = 100;
J = zeros(Nciclos,1);
for ciclo = 1:Nciclos
  # Um ciclo da base inteira:
  sorteio = randperm(length(X));
  for n = sorteio
    # Ida
    y = neu(X(n),w,b);
    # Volta:
    e = (D(n)-y);
    # Ajuste pelo gradiente estocástico (por isso o sorteio)
    w = w + alfa*(1-y)*y*e*X(n);
    b = b + alfa*(1-y)*y*e;
    J(ciclo) = J(ciclo) + e^2;
  end
  J(ciclo) = J(ciclo)/length(sorteio);
end
plot(J)


