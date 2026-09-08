clear;
clc;
close all;

% Mokymo duomenys
x = linspace(0.1, 1, 20);

for n = 1:20
    d(n) = ((1 + 0.6*sin(2*pi*x(n)/0.7)) ...
        + 0.3*sin(2*pi*x(n))) / 2;
end

% Pradiniai svoriai
w1 = randn(1);
w2 = randn(1);
w3 = randn(1);
w4 = randn(1);

b1 = randn(1);
b2 = randn(1);
b3 = randn(1);
b4 = randn(1);

wo1 = randn(1);
wo2 = randn(1);
wo3 = randn(1);
wo4 = randn(1);

bo = randn(1);

% Mokymosi parametrai
eta = 0.2;
iterations = 50000;

% Mokymas
for iteration = 1:iterations

    for n = 1:20

        % Paslėptas sluoksnis
        v1 = x(n) * w1 + b1;
        y1 = 1 / (1 + exp(-v1));

        v2 = x(n) * w2 + b2;
        y2 = 1 / (1 + exp(-v2));

        v3 = x(n) * w3 + b3;
        y3 = 1 / (1 + exp(-v3));

        v4 = x(n) * w4 + b4;
        y4 = 1 / (1 + exp(-v4));

        % Išėjimo neuronas
        y = y1 * wo1 + y2 * wo2 + y3 * wo3 + y4 * wo4 + bo;

        % Klaida
        e = d(n) - y;

        % Backpropagation
        delta_out = e;

        delta1 = y1 * (1 - y1) * delta_out * wo1;
        delta2 = y2 * (1 - y2) * delta_out * wo2;
        delta3 = y3 * (1 - y3) * delta_out * wo3;
        delta4 = y4 * (1 - y4) * delta_out * wo4;

        % Išėjimo sluoksnio svoriai
        wo1 = wo1 + eta * delta_out * y1;
        wo2 = wo2 + eta * delta_out * y2;
        wo3 = wo3 + eta * delta_out * y3;
        wo4 = wo4 + eta * delta_out * y4;

        bo = bo + eta * delta_out;

        % Paslėpto sluoksnio svoriai
        w1 = w1 + eta * delta1 * x(n);
        w2 = w2 + eta * delta2 * x(n);
        w3 = w3 + eta * delta3 * x(n);
        w4 = w4 + eta * delta4 * x(n);

        b1 = b1 + eta * delta1;
        b2 = b2 + eta * delta2;
        b3 = b3 + eta * delta3;
        b4 = b4 + eta * delta4;

    end
end

% Tinklo rezultatas
for n = 1:20

    v1 = x(n) * w1 + b1;
    y1 = 1 / (1 + exp(-v1));

    v2 = x(n) * w2 + b2;
    y2 = 1 / (1 + exp(-v2));

    v3 = x(n) * w3 + b3;
    y3 = 1 / (1 + exp(-v3));

    v4 = x(n) * w4 + b4;
    y4 = 1 / (1 + exp(-v4));

    y_network(n) = y1 * wo1 + y2 * wo2 + y3 * wo3 + y4 * wo4 + bo;

end

% Rezultatai
fprintf('Mokymas baigtas.\n');

fprintf('w1 = %.6f\n', w1);
fprintf('w2 = %.6f\n', w2);
fprintf('w3 = %.6f\n', w3);
fprintf('w4 = %.6f\n', w4);

fprintf('b1 = %.6f\n', b1);
fprintf('b2 = %.6f\n', b2);
fprintf('b3 = %.6f\n', b3);
fprintf('b4 = %.6f\n', b4);

fprintf('wo1 = %.6f\n', wo1);
fprintf('wo2 = %.6f\n', wo2);
fprintf('wo3 = %.6f\n', wo3);
fprintf('wo4 = %.6f\n', wo4);

fprintf('bo = %.6f\n', bo);

% Aproksimacijos grafikas
figure;

plot(x, d, 'o-');
hold on;
plot(x, y_network, 'x-');

xlabel('x');
ylabel('y');
title('Daugiasluoksnio perceptrono aproksimacija');

legend('Norimas atsakas d(x)', 'Tinklo atsakas y(x)');
grid on;