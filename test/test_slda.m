%% sLDA MATLAB simulation
% Replicates the slda.py ROS node output and compares with its CSV output.
%
% Modes:
%   'csv': reads pre-computed FBCSP features (raw mean power), applies log + sLDA.
%   'gdf': runs the full pipeline from a GDF file:
%          CAR → LP→HP (causal, order 4) → ring buffer → CSP → mean(x²) → log → sLDA.
%
% Workflow (CSV):
%   1. roslaunch slda_bci test_node_slda.launch
%   2. Ctrl+C → test/slda_output.csv + test/slda_output_first_seq.txt
%   3. Run this script with input_mode = 'csv'.
%
% Workflow (GDF):
%   1. roslaunch slda_bci test_node_slda_gdf.launch \
%        gdf_file:=$(rospack find processing_bci)/test/prova32ch.gdf
%   2. Ctrl+C → test/slda_gdf_output.csv + test/slda_gdf_output_first_seq.txt
%   3. Run this script with input_mode = 'gdf'.

clear; clc; close all;

%% --- mode ---
input_mode = 'gdf';   % 'csv' | 'gdf'

%% --- paths ---
data_dir  = './test_node_data/';
out_dir   = './test_node_data/slda_bci/';
slda_yaml = './src/slda_bci/models/mi/slda_mi_test.yaml';
car_yaml  = './src/rosneuro_filters_car/cfg/car.yaml';
csp_yaml  = './src/processing_bci/cfg/csp/mi/csp_mi_test.yaml';

if strcmp(input_mode, 'csv')
    fbcsp_file   = [data_dir 'processing_bci/fbcsp_processing.csv'];
    ros_file     = [out_dir  'slda_output.csv'];
    framerate    = 20;
    samplerate   = 500;
    plot_start_s = [];
else
    gdf_file     = [data_dir 'prova32ch.gdf'];
    ros_file     = [out_dir  'slda_gdf_output.csv'];
    framerate    = 16;
    samplerate   = 512;
    plot_start_s = 2;
end

%% --- first_seq ---
first_seq_file = strrep(ros_file, '.csv', '_first_seq.txt');
first_seq = 0;
if isfile(first_seq_file)
    first_seq = readmatrix(first_seq_file);
    fprintf('ROS first_seq = %d\n', first_seq);
else
    fprintf('first_seq file not found — assuming 0.\n');
end

%% --- load sLDA model ---
slda_cfg  = yaml.ReadYaml(slda_yaml);
sp        = slda_cfg.sLDACfg.params;
weights   = cell2mat(sp.slda_weights);     % [1 × n_features]
intercept = cell2mat(sp.slda_intercept);   % scalar
classes   = sp.classes;
n_comp    = numel(sp.selected_components_indices);
n_bands_m = size(cell2mat(sp.bands), 1);
n_feat    = n_bands_m * n_comp;
fprintf('sLDA: %d classes, %d bands × %d comp = %d features\n', numel(classes), n_bands_m, n_comp, n_feat);

%% --- compute reference features ---
if strcmp(input_mode, 'csv')
    raw_features = readmatrix(fbcsp_file);   % [n_frames × n_feat], raw mean power
    n_frames_out = size(raw_features, 1);
    fprintf('CSV: %d frames × %d features\n', n_frames_out, size(raw_features,2));

else  % 'gdf' — full pipeline (mirrors test_fbcsp.m + sLDA)
    car_cfg      = yaml.ReadYaml(car_yaml);
    eog_ch_names = car_cfg.CarCfg.params.EOG_ch_names;
    csp_cfg      = yaml.ReadYaml(csp_yaml);
    csp_params   = csp_cfg.CspCfg.params;

    ncsp_bands   = length(csp_params.csp_matrices);
    csp_matrices = cell(ncsp_bands, 1);
    for b = 1:ncsp_bands
        csp_matrices{b} = cell2mat(csp_params.csp_matrices{b});
    end
    ncomponents = size(csp_matrices{1}, 1);
    bands_csp   = cell2mat(csp_params.bands);   % [ncsp_bands × 2]
    filterOrder = 4;

    csp_ch_names = {};
    if isfield(csp_params, 'selected_channels')
        csp_ch_names = csp_params.selected_channels;
    end

    [data_raw, hdr] = sload(gdf_file);   % BIOSIG required
    sr        = hdr.SampleRate;
    ch_names  = cellstr(hdr.Label);
    n_eeg     = sum(~cellfun(@(c) contains(lower(c), {'status','trigger','mkr'}), ch_names));
    data      = data_raw(:, 1:n_eeg);
    ch_names  = ch_names(1:n_eeg);
    fprintf('GDF: %d samples × %d channels @ %.0f Hz\n', size(data,1), n_eeg, sr);

    nchannels  = size(data, 2);
    chunkSize  = round(sr / framerate);
    bufferSize = round(sr);

    % Resolve EOG channels
    EOG_ch = zeros(1, numel(eog_ch_names));
    for k = 1:numel(eog_ch_names)
        m = find(strcmpi(ch_names, eog_ch_names{k}), 1);
        if isempty(m), error('EOG channel "%s" not found.', eog_ch_names{k}); end
        EOG_ch(k) = m;
    end
    non_eog_ch = setdiff(1:nchannels, EOG_ch);

    % Resolve CSP channel subset
    if isempty(csp_ch_names)
        csp_ch = 1:nchannels;
    else
        csp_ch = zeros(1, numel(csp_ch_names));
        for k = 1:numel(csp_ch_names)
            m = find(strcmpi(ch_names, csp_ch_names{k}), 1);
            if isempty(m), error('CSP channel "%s" not found.', csp_ch_names{k}); end
            csp_ch(k) = m;
        end
    end

    % Design causal LP→HP filters (order 4, matching Fbcsp.cpp)
    nyq = sr / 2;
    b_lp = cell(ncsp_bands,1); a_lp = cell(ncsp_bands,1);
    b_hp = cell(ncsp_bands,1); a_hp = cell(ncsp_bands,1);
    zi_lp = cell(ncsp_bands,1); zi_hp = cell(ncsp_bands,1);
    for b = 1:ncsp_bands
        [b_lp{b}, a_lp{b}] = butter(filterOrder, bands_csp(b,2)/nyq, 'low');
        [b_hp{b}, a_hp{b}] = butter(filterOrder, bands_csp(b,1)/nyq, 'high');
        zi_lp{b} = zeros(max(length(a_lp{b}), length(b_lp{b}))-1, nchannels);
        zi_hp{b} = zeros(max(length(a_hp{b}), length(b_hp{b}))-1, nchannels);
    end

    % Ring buffers (NaN = not yet full, mirrors C++ RingBuffer)
    bufs = nan(bufferSize, nchannels, ncsp_bands);

    n_frames     = floor(size(data,1) / chunkSize);
    raw_features = zeros(n_frames, ncomponents * ncsp_bands);

    for seq = 0 : n_frames - 1
        f   = seq + 1;
        idx = seq * chunkSize + 1 : (seq+1) * chunkSize;
        chunk = data(idx, :);

        % CAR
        car_mean  = mean(chunk(:, non_eog_ch), 2);
        chunk_car = chunk - car_mean;

        % LP → HP per band, update ring buffer
        for b = 1:ncsp_bands
            [lp_out, zi_lp{b}] = filter(b_lp{b}, a_lp{b}, chunk_car, zi_lp{b}, 1);
            [bp_out, zi_hp{b}] = filter(b_hp{b}, a_hp{b}, lp_out,    zi_hp{b}, 1);
            bufs(:,:,b) = [bufs(chunkSize+1:end,:,b); bp_out];
        end

        if any(isnan(bufs(:)))
            % Buffer not full — output ones (matches ROS behavior for warmup frames)
            raw_features(f, :) = ones(1, ncomponents * ncsp_bands);
            continue;
        end

        % CSP + mean(x²) per component
        csp_feats = zeros(ncomponents, ncsp_bands);
        for b = 1:ncsp_bands
            buf_sel = bufs(:, csp_ch, b);
            csp_out = buf_sel * csp_matrices{b}';
            csp_feats(:, b) = sum(csp_out.^2, 1)' / bufferSize;
        end
        % Column-major flatten: [comp1_band1, ..., compN_band1, comp1_band2, ...]
        raw_features(f, :) = reshape(csp_feats, 1, []);
    end
    n_frames_out = n_frames;
    fprintf('GDF: %d frames processed (chunkSize=%d, bufferSize=%d)\n', n_frames_out, chunkSize, bufferSize);
end

%% --- apply log + sLDA ---
log_feats  = log(raw_features);               % [n_frames × n_feat]
scores     = log_feats * weights' + intercept; % [n_frames × 1]
probs_c2   = 1 ./ (1 + exp(-scores));
probs_c1   = 1 - probs_c2;
matlab_out = [probs_c1, probs_c2];            % [n_frames × 2]

%% --- load ROS output ---
if ~isfile(ros_file)
    warning('ROS output not found: %s', ros_file);
    return;
end
ros_data = readmatrix(ros_file);   % [n_ros_frames × 2]
fprintf('ROS output: %d frames\n', size(ros_data, 1));

ch_ref = 2;   % compare class-2 probability column

n_cmp   = min(size(ros_data,1), size(matlab_out,1));
ros_out = ros_data(1:n_cmp, :);
mat_cmp = matlab_out(1:n_cmp, :);

% Align by first_seq
skip    = min(first_seq + 1, n_cmp);   % 1-based
ros_cmp = ros_out(skip:end, :);
mat_cmp = mat_cmp(skip:end, :);

%% --- xcorr alignment ---
MAX_LAG = 20;
n_xcorr = min(size(ros_cmp,1), size(mat_cmp,1));
r_ref = ros_cmp(1:n_xcorr, ch_ref) - mean(ros_cmp(1:n_xcorr, ch_ref));
m_ref = mat_cmp(1:n_xcorr, ch_ref) - mean(mat_cmp(1:n_xcorr, ch_ref));

[xcf, lags] = xcorr(r_ref, m_ref, MAX_LAG, 'normalized');
[~, peak_idx] = max(xcf);
lag = lags(peak_idx);
fprintf('Cross-corr lag: %+d frame(s)  ', lag);
if lag == 0
    fprintf('[no residual lag]\n');
elseif lag > 0
    fprintf('[ROS lags MATLAB]\n');
else
    fprintf('[MATLAB lags ROS]\n');
end

if lag > 0
    r_aligned = ros_cmp(1+lag:end, :);
    m_aligned = mat_cmp(1:end-lag, :);
elseif lag < 0
    shift     = -lag;
    r_aligned = ros_cmp(1:end-shift, :);
    m_aligned = mat_cmp(1+shift:end, :);
else
    r_aligned = ros_cmp;
    m_aligned = mat_cmp;
end

%% --- restrict plot window ---
fr = framerate;
if ~isempty(plot_start_s)
    aligned_skip = min(round(plot_start_s * fr), size(r_aligned,1));
else
    aligned_skip = 0;
end
r_al_plot = r_aligned(aligned_skip+1:end, :);
m_al_plot = m_aligned(aligned_skip+1:end, :);
t_raw = (0 : n_xcorr - 1) / fr;
t_al  = (aligned_skip : aligned_skip + size(r_al_plot,1) - 1) / fr;

mae_raw     = mean(abs(r_ref - m_ref));
mae_aligned = mean(abs(r_al_plot(:,ch_ref) - m_al_plot(:,ch_ref)));
if ~isempty(plot_start_s)
    fprintf('Plotting from %.1f s onward (%d aligned frames)\n', plot_start_s, size(r_al_plot,1));
end
fprintf('MAE p(c2) [raw]     : %.6f\n', mae_raw);
fprintf('MAE p(c2) [aligned] : %.6f\n', mae_aligned);

%% --- plot: raw ---
figure;
subplot(2,1,1); hold on;
plot(t_raw, ros_cmp(1:n_xcorr, ch_ref), 'b',   'LineWidth', 1.5);
plot(t_raw, mat_cmp(1:n_xcorr, ch_ref), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('p(class 2)'); grid on; hold off;
title(sprintf('[RAW] sLDA | first\\_seq=%d | mode=%s', first_seq, upper(input_mode)));
subplot(2,1,2);
bar(t_raw, abs(ros_cmp(1:n_xcorr, ch_ref) - mat_cmp(1:n_xcorr, ch_ref)));
xlabel('time [s]'); ylabel('|diff|');
title(sprintf('Differences (lag=%+d frames)', lag)); grid on;

%% --- plot: lag-corrected ---
figure;
subplot(2,1,1); hold on;
plot(t_al, r_al_plot(:,ch_ref), 'b',   'LineWidth', 1.5);
plot(t_al, m_al_plot(:,ch_ref), 'r--', 'LineWidth', 1);
legend('ROS node', 'MATLAB simulation');
ylabel('p(class 2)'); grid on; hold off;
title(sprintf('[ALIGNED lag=%+d] sLDA | p(c2) | mode=%s', lag, upper(input_mode)));
subplot(2,1,2);
bar(t_al, abs(r_al_plot(:,ch_ref) - m_al_plot(:,ch_ref)));
xlabel('time [s]'); ylabel('|diff|');
title('Differences after alignment'); grid on;
