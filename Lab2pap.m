% Iejimu reiksmes
x1 = 0:0.05:1;
x2 = 0:0.05:1;

% Mokymosi greitis
eta = 0.1;

% Iteraciju skaicius
iteracijos = 80000;

% Svoriai
w11 = randn * 0.5;
w12 = randn * 0.5;
w21 = randn * 0.5;
w22 = randn * 0.5;
w31 = randn * 0.5;
w32 = randn * 0.5;
w41 = randn * 0.5;
w42 = randn * 0.5;

% Paslepto sluoksnio poslinkiai
b1 = 0;
b2 = 0;
b3 = 0;
b4 = 0;

% Isejimo svoriai
wo1 = randn * 0.5;
wo2 = randn * 0.5;
wo3 = randn * 0.5;
wo4 = randn * 0.5;

% Isejimo poslinkis
bo = 0;

% Tinklo mokymas
for iteracija = 1:iteracijos

    for i = 1:length(x1)

        for j = 1:length(x2)

            % Apskaiciuojama norima reiksme
            d = (sin(2*pi*x1(i)) + cos(2*pi*x2(j)) + 2) / 4;

            % Paslepto sluoksnio neuronai
            v1 = w11*x1(i) + w12*x2(j) + b1;
            v2 = w21*x1(i) + w22*x2(j) + b2;
            v3 = w31*x1(i) + w32*x2(j) + b3;
            v4 = w41*x1(i) + w42*x2(j) + b4;

            % Sigmoides funkcija
            h1 = 1/(1+exp(-v1));
            h2 = 1/(1+exp(-v2));
            h3 = 1/(1+exp(-v3));
            h4 = 1/(1+exp(-v4));

            % Apskaiciuojamas tinklo isejimas
            y = wo1*h1 + wo2*h2 + wo3*h3 + wo4*h4 + bo;

            % Apskaiciuojama klaida
            e = d - y;

            % Backpropagation
            delta_o = e;

            delta1 = delta_o * wo1 * h1*(1-h1);
            delta2 = delta_o * wo2 * h2*(1-h2);
            delta3 = delta_o * wo3 * h3*(1-h3);
            delta4 = delta_o * wo4 * h4*(1-h4);

            % Atnaujinami isejimo svoriai
            wo1 = wo1 + eta*delta_o*h1;
            wo2 = wo2 + eta*delta_o*h2;
            wo3 = wo3 + eta*delta_o*h3;
            wo4 = wo4 + eta*delta_o*h4;

            bo = bo + eta*delta_o;

            % Atnaujinami paslepto sluoksnio svoriai
            w11 = w11 + eta*delta1*x1(i);
            w12 = w12 + eta*delta1*x2(j);

            w21 = w21 + eta*delta2*x1(i);
            w22 = w22 + eta*delta2*x2(j);

            w31 = w31 + eta*delta3*x1(i);
            w32 = w32 + eta*delta3*x2(j);

            w41 = w41 + eta*delta4*x1(i);
            w42 = w42 + eta*delta4*x2(j);

            % Atnaujinami poslinkiai
            b1 = b1 + eta*delta1;
            b2 = b2 + eta*delta2;
            b3 = b3 + eta*delta3;
            b4 = b4 + eta*delta4;

        end
    end
end

% Apskaiciuojamas aproksimuotas pavirsius
n = 1;

for i = 1:length(x1)

    for j = 1:length(x2)

        v1 = w11*x1(i) + w12*x2(j) + b1;
        v2 = w21*x1(i) + w22*x2(j) + b2;
        v3 = w31*x1(i) + w32*x2(j) + b3;
        v4 = w41*x1(i) + w42*x2(j) + b4;

        h1 = 1/(1+exp(-v1));
        h2 = 1/(1+exp(-v2));
        h3 = 1/(1+exp(-v3));
        h4 = 1/(1+exp(-v4));

        Y(n) = wo1*h1 + wo2*h2 + wo3*h3 + wo4*h4 + bo;

        n = n + 1;
    end
end

% Paruosiamas pavirsius braizymui
[X1,X2] = meshgrid(x1,x2);

% Apskaiciuojamas norimas pavirsius
D = (sin(2*pi*X1) + cos(2*pi*X2) + 2) / 4;

% Pakeiciamas Y masyvo formatas
Y = reshape(Y,length(x2),length(x1));

% Braizomas norimas pavirsius
figure;
surf(X1,X2,D);
title('Norimas pavirsius');
xlabel('x_1');
ylabel('x_2');
zlabel('d');

% Braizomas aproksimuotas pavirsius
figure;
surf(X1,X2,Y);
title('Neuroninio tinklo aproksimuotas pavirsius');
xlabel('x_1');
ylabel('x_2');
zlabel('y');
