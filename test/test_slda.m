clear; clc; close all;

datapath = './src/slda_bci/';

path_raw_csv   = [datapath 'test/fbcsp_processing.csv'];
path_ros_csv   = [datapath 'test/slda_output.csv'];
path_slda_yaml = [datapath '/models/slda_test.yaml'];

fprintf('Loading data...\n');

raw_data = readmatrix(path_raw_csv);
ros_data = readmatrix(path_ros_csv);


model = yaml.ReadYaml(path_slda_yaml);
weights = cell2mat(model.sLDACfg.params.slda_weights);
intercept = cell2mat(model.sLDACfg.params.slda_intercept);
fprintf('Model loaded correctly from YAML.\n');



%% --- MATLAB Implementation of sLDA ---
dfet = log(raw_data);

% lda
scores = dfet * weights' + intercept;
mat_probs_c2 = 1 ./ (1 + exp(-scores));
mat_probs_c1 = 1 - mat_probs_c2;
matlab_output = [mat_probs_c1, mat_probs_c2];

%% --- Comparison and visualization ---
ros_skip_msgs = 0; 
align =  + 40;

ros_aligned = ros_data(align:end, :);
matlab_aligned = matlab_output(align + ros_skip_msgs:end, :);

n_samples = min(size(matlab_aligned, 1), size(ros_aligned, 1));

ros_final = ros_aligned(1:n_samples, :);
matlab_final = matlab_aligned(1:n_samples, :);

diff = matlab_final - ros_final;
mse = mean(diff.^2, 'all');

fprintf('Comparison Results:\n');
fprintf('  Analyzed Samples: %d\n', n_samples);
fprintf('  Mean Squared Error (MSE): %e\n', mse);
fprintf('  Max Absolute Error: %e\n', max(abs(diff), [], 'all'));

% --- Plotting ---
figure('Name', 'sLDA Parity Check: ROS vs MATLAB', 'Color', 'w');

% Subplot 1: Comparazione Probabilità (Classe 2)
subplot(2,1,1);
plot(matlab_final(:, 2), 'LineWidth', 2, 'DisplayName', 'MATLAB (Class 2)');
hold on;
plot(ros_final(:, 2), '--', 'LineWidth', 2, 'DisplayName', 'ROS (Class 2)');
grid on;
title('Probability Comparison (Class 2)');
xlabel('Sample Index (Aligned)');
ylabel('Probability');
legend('Location', 'best');

% Subplot 2: Differenza puntuale
subplot(2,1,2);
plot(diff(:, 2), 'r', 'LineWidth', 1.5);
grid on;
title('Difference (MATLAB - ROS)');
xlabel('Sample Index (Aligned)');
ylabel('\Delta Prob');

% Messaggio di validazione finale
if mse < 1e-10
    fprintf('SUCCESS: The implementations are mathematically equivalent!\n');
else
    fprintf('WARNING: Discrepancy detected. Check feature ordering, buffer alignment or normalization.\n');
end

sgtitle('Evaluation sLDA: ROS node simulation vs MATLAB');
