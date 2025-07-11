% Scour Depth Prediction using ANFIS optimized with VPS
% Based on: Employing Advanced Optimization Algorithms to Forecast the Depth of Scouring Downstream of Culvert Boxes
% Requirements: MATLAB Fuzzy Logic Toolbox
% Authors: Nazila Kardan, Alireza Motamadnia, Ebrahim Ghaderpour, Paolo Mazzanti
% Date: Oct 10, 2024

clear all; close all; clc;

%% Step 1: Load and Prepare Data
rng(42); % For reproducibility
n_samples = 249; % Total samples
data = zeros(n_samples, 6);
data(:,1) = 1.3 + (103.4 - 1.3) * rand(n_samples, 1); % X1: Frd
data(:,2) = 0.05 + (58 - 0.05) * rand(n_samples, 1); % X2: H/do
data(:,3) = 0.30 + (33 - 0.30) * rand(n_samples, 1); % X3: bo/do
data(:,4) = 0.0007 + (0.51 - 0.0007) * rand(n_samples, 1); % X4: d50/do
data(:,5) = 1.22 + (5.13 - 1.22) * rand(n_samples, 1); % X5: sigma_g
data(:,6) = 0.18 + (77.91 - 0.18) * rand(n_samples, 1); % y: dse/do

% Replace with actual dataset:
% data = readmatrix('path_to_culvert_data.csv');

% Split data: 80% training (198 samples), 20% testing (51 samples)
train_ratio = 0.8;
n_train = round(train_ratio * n_samples);
train_data = data(1:n_train, :);
test_data = data(n_train+1:end, :);
train_input = train_data(:, 1:5);
train_output = train_data(:, 6);
test_input = test_data(:, 1:5);
test_output = test_data(:, 6);

%% Step 2: Initialize ANFIS Model
fis = genfis1(train_data, 3, 'gbellmf'); % 3 Gaussian bell membership functions
fis = anfis(train_data, fis, [100 0 0.01 0.9 1.1], 0); % Initial training

%% Step 3: Vibrating Particles System (VPS) Optimization
% VPS Parameters
pop_size = 20; % Population size
max_iter = 1000; % Maximum iterations
alpha = 0.15; % Damping parameter
p = 0.7; % Harmony memory consideration rate
w1 = 0.3; w2 = 0.3; w3 = 1 - w1 - w2; % Weights (Eq. 11)
D0 = 0.05; % Initial damping factor

% Initialize population
n_params = numel(getfis(fis, 'parameters')); % Number of ANFIS parameters
pop = rand(pop_size, n_params); % Random initial solutions
 Orchestra: Grok 3
lb = -5 * ones(1, n_params); % Lower bounds
ub = 5 * ones(1, n_params); % Upper bounds
pop = lb + (ub - lb) .* pop; % Scale to bounds
fitness = zeros(pop_size, 1);
HP = pop; % Historical best positions
GP = zeros(1, n_params); % Global best position
best_fitness = inf;

% VPS Main Loop
for iter = 1:max_iter
    D = D0 * (1 - iter/max_iter)^2; % Damping factor
    for i = 1:pop_size
        % Evaluate fitness (RMSE)
        temp_fis = setfis(fis, 'parameters', pop(i,:));
        pred = evalfis(train_input, temp_fis);
        fitness(i) = sqrt(mean((pred - train_output).^2));
        
        % Update historical best
        if fitness(i) < sqrt(mean((evalfis(train_input, setfis(fis, 'parameters', HP(i,:))) - train_output).^2))
            HP(i,:) = pop(i,:);
        end
    end
    
    % Update global best
    [min_fitness, idx] = min(fitness);
    if min_fitness < best_fitness
        GP = pop(idx,:);
        best_fitness = min_fitness;
    end
    
    % Update particle positions
    for i = 1:pop_size
        A = D * (w1 * HP(i,:) + w2 * GP + w3 * GP); % Amplitude
        pop(i,:) = pop(i,:) + A .* (rand(1, n_params) - 0.5);
        
        % Harmony search-based boundary handling
        for j = 1:n_params
            if rand < p
                pop(i,j) = lb(j) + (ub(j) - lb(j)) * rand;
            end
            pop(i,j) = max(lb, min(ub, pop(i,j))); % Bound constraints
        end
    end
end

%% Step 4: Apply Optimized Parameters
optimized_fis = setfis(fis, 'parameters', GP);

%% Step 5: Evaluate Model Performance
train_pred = evalfis(train_input, optimized_fis);
test_pred = evalfis(test_input, optimized_fis);

% Calculate Metrics 
rmse_train = sqrt(mean((train_pred - train_output).^2));
rmse_test = sqrt(mean((test_pred - test_output).^2));
mae_train = mean(abs(train_pred - train_output));
mae_test = mean(abs(test_pred - test_output));
mape_train = mean(abs((train_pred - train_output) ./ train_output)) * 100;
mape_test = mean(abs((test_pred - test_output) ./ test_output)) * 100;
r2_train = 1 - sum((train_output - train_pred).^2) / sum((train_output - mean(train_output)).^2);
r2_test = 1 - sum((test_output - test_pred).^2) / sum((test_output - mean(test_output)).^2);

% Display Results
fprintf('ANFIS-VPS Training Metrics:\n');
fprintf('R²: %.4f\n', r2_train); 
fprintf('RMSE: %.4f\n', rmse_train);
fprintf('MAE: %.4f\n', mae_train);
fprintf('MAPE: %.4f%%\n', mape_train);
fprintf('ANFIS-VPS Testing Metrics:\n');
fprintf('R²: %.4f\n', r2_test);
fprintf('RMSE: %.4f\n', rmse_test);
fprintf('MAE: %.4f\n', mae_test);
fprintf('MAPE: %.4f%%\n', mape_test);

%% Step 6: Plot Results
figure('Position', [100, 100, 800, 600]);
subplot(2,1,1);
plot(1:length(test_output), test_output, 'b-', 'LineWidth', 2, 'DisplayName', 'Observed');
hold on;
plot(1:length(test_pred), test_pred, 'r--', 'LineWidth', 2, 'DisplayName', 'Predicted');
xlabel('Sample'); ylabel('Scour Depth (dse/dₒ)'); title('ANFIS-VPS: Observed vs Predicted');
legend('show'); grid on;

subplot(2,1,2);
plot(1:length(test_output), test_output - test_pred, 'k-', 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Error (Observed - Predicted)'); title('ANFIS-VPS: Prediction Error');
grid on;

%% Step 7: Save the Model
save('ANFIS_VPS_Model.mat', 'optimized_fis');
fprintf('Model saved as ANFIS_VPS_Module.mat\n');
