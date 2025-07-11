% Scour Depth Prediction using ANFIS optimized with CBO
% Based on: Employing Advanced Optimization Algorithms to Forecast the Depth of Scouring Downstream of Culvert Boxes
% Authors: Nazila Kardan, Alireza Motamadnia, Ebrahim Ghaderpour, Paolo Mazzanti
% Date: July 10, 2024

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

%% Step 3: Colliding Bodies Optimization (CBO)
% CBO Parameters 
pop_size = 20; % Number of colliding bodies
max_iter = 1000; % Maximum iterations
n_params = numel(getfis(fis, 'parameters')); % Number of ANFIS parameters
pop = rand(pop_size, n_params); % Random initial solutions
lb = -5 * ones(1, n_params); % Lower bounds
ub = 5 * ones(1, n_params); % Upper bounds
pop = lb + (ub - lb) .* pop; % Scale to bounds
fitness = zeros(pop_size, 1);
best_fitness = inf;
best_solution = zeros(1, n_params);

% CBO Main Loop
for iter = 1:max_iter
    % Calculate fitness (RMSE)
    for i = 1:pop_size
        temp_fis = setfis(fis, 'parameters', pop(i,:));
        pred = evalfis(train_input, temp_fis);
        fitness(i) = sqrt(mean((pred - train_output).^2));
    end
    
    % Update best solution
    [min_fitness, idx] = min(fitness);
    if min_fitness < best_fitness
        best_fitness = min_fitness;
        best_solution = pop(idx,:);
    end
    
    % Sort bodies by fitness
    [~, sorted_idx] = sort(fitness);
    pop = pop(sorted_idx, :);
    fitness = fitness(sorted_idx);
    
    % Calculate masses 
    masses = 1 ./ (fitness + eps); % Avoid division by zero
    masses = masses / sum(masses); % Normalize
    
    % Update velocities and positions
    v = zeros(pop_size, n_params); % Initial velocities
    for i = 1:pop_size/2
        % Pair bodies (best with worst, second best with second worst, etc.)
        i1 = i; % Stationary body
        i2 = pop_size - i + 1; % Moving body
        v_pre = v(i2,:); % Velocity before collision
        COR = 1 - iter/max_iter; % Coefficient of restitution
        
        % Post-collision velocity for moving body
        v(i2,:) = (masses(i1) * v_pre) / (masses(i1) + masses(i2));
        v(i1,:) = (masses(i2) * v_pre * COR) / (masses(i1) + masses(i2));
        
        % Update positions
        pop(i1,:) = pop(i1,:) + rand(1, n_params) .* v(i1,:);
        pop(i2,:) = pop(i2,:) + rand(1, n_params) .* v(i2,:);
        
        % Bound constraints
        pop(i1,:) = max(lb, min(ub, pop(i1,:)));
        pop(i2,:) = max(lb, min(ub, pop(i2,:)));
    end
end

%% Step 4: Apply Optimized Parameters
optimized_fis = setfis(fis, 'parameters', best_solution);

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
fprintf('ANFIS-CBO Training Metrics:\n');
fprintf('R²: %.4f\n', r2_train);
fprintf('RMSE: %.4f\n', rmse_train);
fprintf('MAE: %.4f\n', mae_train);
fprintf('MAPE: %.4f%%\n', mape_train);
fprintf('ANFIS-CBO Testing Metrics:\n');
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
xlabel('Sample'); ylabel('Scour Depth (dse/dₒ)'); title('ANFIS-CBO: Observed vs Predicted');
legend('show'); grid on;

subplot(2,1,2);
plot(1:length(test_output), test_output - test_pred, 'k-', 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Error (Observed - Predicted)'); title('ANFIS-CBO: Prediction Error');
grid on;

%% Step 7: Save the Model
save('ANFIS_CBO_Model.mat', 'optimized_fis');
fprintf('Model saved as ANFIS_CBO_Model.mat\n');
