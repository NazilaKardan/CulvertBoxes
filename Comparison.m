% Comparison of ANFIS-CBO, ANFIS-ECBO, and ANFIS-VPS for Scour Depth Prediction
% Based on: Employing Advanced Optimization Algorithms to Forecast the Depth of Scouring Downstream of Culvert Boxes
% Authors: Nazila Kardan, Alireza Motamadnia, Ebrahim Ghaderpour, Paolo Mazzanti
% Date: OCT 22, 2024

clear all; close all; clc;

%% Step 1: Load Test Data
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

% Use test data (20% of dataset)
train_ratio = 0.8;
n_train = round(train_ratio * n_samples);
test_data = data(n_train+1:end, :);
test_input = test_data(:, 1:5);
test_output = test_data(:, 6);

%% Step 2: Load Saved Models
load('ANFIS_CBO_Model.mat', 'optimized_fis');
fis_cbo = optimized_fis;
load('ANFIS_ECBO_Model.mat', 'optimized_fis');
fis_ecbo = optimized_fis;
load('ANFIS_VPS_Model.mat', 'optimized_fis');
fis_vps = optimized_fis;

%% Step 3: Generate Predictions
pred_cbo = evalfis(test_input, fis_cbo);
pred_ecbo = evalfis(test_input, fis_ecbo);
pred_vps = evalfis(test_input, fis_vps);

%% Step 4: Calculate Evaluation Metrics
% CBO Metrics
rmse_cbo = sqrt(mean((pred_cbo - test_output).^2));
mae_cbo = mean(abs(pred_cbo - test_output));
mape_cbo = mean(abs((pred_cbo - test_output) ./ test_output)) * 100;
r2_cbo = 1 - sum((test_output - pred_cbo).^2) / sum((test_output - mean(test_output)).^2);

% ECBO Metrics
rmse_ecbo = sqrt(mean((pred_ecbo - test_output).^2));
mae_ecbo = mean(abs(pred_ecbo - test_output));
mape_ecbo = mean(abs((pred_ecbo - test_output) ./ test_output)) * 100;
r2_ecbo = 1 - sum((test_output - pred_ecbo).^2) / sum((test_output - mean(test_output)).^2);

% VPS Metrics
rmse_vps = sqrt(mean((pred_vps - test_output).^2));
mae_vps = mean(abs(pred_vps - test_output));
mape_vps = mean(abs((pred_vps - test_output) ./ test_output)) * 100;
r2_vps = 1 - sum((test_output - pred_vps).^2) / sum((test_output - mean(test_output)).^2);

%% Step 5: Display Comparison Table
fprintf('Comparison of ANFIS Models (Testing Metrics):\n');
fprintf('------------------------------------------------\n');
fprintf('Model\t\tR²\tRMSE\tMAE\tMAPE\n');
fprintf('------------------------------------------------\n');
fprintf('ANFIS-CBO\t%.4f\t%.4f\t%.4f\t%.4f%%\n', r2_cbo, rmse_cbo, mae_cbo, mape_cbo);
fprintf('ANFIS-ECBO\t%.4f\t%.4f\t%.4f\t%.4f%%\n', r2_ecbo, rmse_ecbo, mae_ecbo, mape_ecbo);
fprintf('ANFIS-VPS\t%.4f\t%.4f\t%.4f\t%.4f%%\n', r2_vps, rmse_vps, mae_vps, mape_vps);
fprintf('------------------------------------------------\n');

%% Step 6: Plot Comparison
% Scatter Plot: Observed vs Predicted
figure('Position', [100, 100, 800, 600]);
subplot(2,1,1);
plot(test_output, test_output, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Ideal');
hold on;
plot(test_output, pred_cbo, 'bo', 'MarkerSize', 6, 'DisplayName', 'ANFIS-CBO');
plot(test_output, pred_ecbo, 'r+', 'MarkerSize', 6, 'DisplayName', 'ANFIS-ECBO');
plot(test_output, pred_vps, 'g*', 'MarkerSize', 6, 'DisplayName', 'ANFIS-VPS');
xlabel('Observed Scour Depth (dse/dₒ)'); ylabel('Predicted Scour Depth (dse/dₒ)');
title('Comparison of Observed vs Predicted Scour Depth');
legend('show'); grid on;

% Bar Plot: Evaluation Metrics
subplot(2,1,2);
metrics = [r2_cbo, r2_ecbo, r2_vps; rmse_cbo, rmse_ecbo, rmse_vps; ...
           mae_cbo, mae_ecbo, mae_vps; mape_cbo, mape_ecbo, mape_vps];
bar(metrics');
set(gca, 'XTickLabel', {'R²', 'RMSE', 'MAE', 'MAPE'});
legend({'ANFIS-CBO', 'ANFIS-ECBO', 'ANFIS-VPS'}, 'Location', 'best');
ylabel('Metric Value'); title('Comparison of Evaluation Metrics');
grid on;

%% Step 7: Save Comparison Results
save('comparison_results.mat', 'r2_cbo', 'rmse_cbo', 'mae_cbo', 'mape_cbo', ...
     'r2_ecbo', 'rmse_ecbo', 'mae_ecbo', 'mape_ecbo', ...
     'r2_vps', 'rmse_vps', 'mae_vps', 'mape_vps');
fprintf('Comparison results saved as comparison_results.mat\n');
