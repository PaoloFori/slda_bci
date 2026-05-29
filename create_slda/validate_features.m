%% VALIDATE_FEATURES  Compare Python training features (create_slda.ipynb last cell)
%   against MATLAB apply_processing output on the same calibration GDF(s).
%
%   Selects one or more _training_features.mat files (one per GDF).  For each:
%     1. Loads the GDF at the path stored in py.gdf_path.
%     2. Builds a CSP struct from the saved csp_matrices.
%     3. Runs apply_processing (CAR + ba-form Butterworth + ring buffer + CSP).
%     4. Extracts CF-window features: k_ml = pos_k + j_in_trial + 1.
%     5. Applies log and compares with X_py.
%
%   Python: sosfilt whole-recording -> window extraction -> MNE CSP.transform()
%   MATLAB: ba-form Butterworth chunk-by-chunk -> ring buffer -> CSP -> mean(x^2)
%   Expected MAE < 1e-5  (filter-form numerical difference only).
%
%   Array layout notes (scipy preserves shape for N-D arrays):
%     py.X_py        MATLAB sees [N_win, N_BANDS, N_CSP_COMP]  (same as numpy)
%     py.csp_matrices MATLAB sees [N_BANDS, N_CSP_COMP, n_sel]  (same as numpy)

clear; clc; close all;

% ── Path setup ────────────────────────────────────────────────────────────────
this_dir = fileparts(mfilename('fullpath'));
parts    = strsplit(this_dir, filesep);
ws_root  = '';
for k = numel(parts):-1:2
    candidate = strjoin(parts(1:k), filesep);
    if isfolder(fullfile(candidate, 'src', 'analysis_bci', 'matlab_simulation'))
        ws_root = candidate; break;
    end
end
if isempty(ws_root)
    error('validate_features:ws', 'Cannot find workspace root from %s', this_dir);
end
sim_dir = fullfile(ws_root, 'src', 'analysis_bci', 'matlab_simulation');
addpath(sim_dir, fullfile(sim_dir,'io'), fullfile(sim_dir,'processing'), ...
        fullfile(sim_dir,'utils'));

% ── Select _training_features.mat file(s) ─────────────────────────────────────
default_dir = fullfile(ws_root, 'recordings');
if ~isfolder(default_dir), default_dir = ws_root; end

[mat_names, mat_dir] = uigetfile({'*.mat','MAT files (*.mat)'}, ...
    'Select _training_features.mat file(s)', default_dir, 'MultiSelect', 'on');
if isequal(mat_names, 0)
    error('validate_features:cancel', 'No file selected.');
end
if ischar(mat_names), mat_names = {mat_names}; end

CF_CODE  = 781;
all_py   = [];  % accumulate Python flat log-features across files
all_ml   = [];  % accumulate MATLAB flat log-features across files
n_files  = numel(mat_names);

% Saved from first file for trace diagnostic plot
trace1.feat_ml_all = [];   % full MATLAB log-feature trace [n_chunks, n_feat]
trace1.feat_ml_log = [];   % aligned MATLAB log-features [N_win, n_feat]
trace1.X_py_flat   = [];   % Python features [N_win, n_feat]
trace1.k_ml_arr    = [];   % MATLAB chunk index for each window [N_win]
trace1.valid       = [];
trace1.n_bands     = 0;
trace1.n_comp      = 0;
trace1.bands       = [];

for fi = 1:n_files
    mat_path = fullfile(mat_dir, mat_names{fi});
    fprintf('\n[%d/%d] %s\n', fi, n_files, mat_names{fi});

    py      = load(mat_path);
    n_bands = round(py.n_bands);
    n_comp  = round(py.n_csp_comp);
    j_arr   = round(py.j_in_trial(:));  % [N_win] 0-based window within trial
    t_arr   = round(py.trial_idx(:));   % [N_win] 0-based trial index in file
    N_win   = numel(j_arr);

    % py.X_py: scipy preserves shape -> MATLAB sees [N_win, N_BANDS, N_CSP_COMP]
    % Flatten to band-major [N_win, N_BANDS*N_CSP_COMP]:
    %   permute -> [N_win, N_CSP_COMP, N_BANDS], then Fortran reshape (comp fast, band slow)
    X_py_flat = reshape(permute(py.X_py, [1, 3, 2]), N_win, n_comp * n_bands);
    fprintf('  Python windows: %d   features: %d (%d bands x %d comp)\n', ...
            N_win, size(X_py_flat, 2), n_bands, n_comp);

    % py.X_pre_csp: scipy preserves shape -> MATLAB sees [N_win, N_BANDS, n_sel_ch]
    % band-major 2-D: [N_win, N_BANDS * n_sel_ch] matching features_pre layout
    n_sel_ch    = size(py.X_pre_csp, 3);
    X_pre_py    = reshape(py.X_pre_csp, N_win, n_bands * n_sel_ch);   % band-major

    % ── Build CSP struct ──────────────────────────────────────────────────────
    % py.csp_matrices: scipy preserves shape -> MATLAB sees [N_BANDS, N_CSP_COMP, n_sel]
    % For band b: squeeze(M(b,:,:)) gives [N_CSP_COMP, n_sel] directly (no transpose needed)
    csp_struct.n_bands           = n_bands;
    csp_struct.n_components      = n_comp;
    csp_struct.bands             = py.bands;           % [n_bands, 2]
    csp_struct.selected_channels = cellstr(py.selected_channels);
    csp_struct.csp_matrices      = cell(n_bands, 1);
    for b = 1:n_bands
        csp_struct.csp_matrices{b} = squeeze(py.csp_matrices(b, :, :));  % [N_CSP_COMP, n_sel]
    end

    % ── Load GDF + companion YAML ─────────────────────────────────────────────
    gdf_path = char(py.gdf_path);
    if ~isfile(gdf_path)
        fprintf('  [WARN] GDF not found: %s — skipping.\n', gdf_path);
        continue;
    end
    [signal, header, ~] = load_gdf(gdf_path);

    try
        [params, ~] = load_params_yaml(gdf_path);
        fs         = double(params.acquisition.samplerate);
        framerate  = double(params.acquisition.framerate);
        chunk_size = round(fs / framerate);
        if abs(fs - header.SampleRate) > 1e-3, fs = header.SampleRate; end
        bufsize    = double(params.RingBufferCfg.params.size);
        eog_names  = to_strcell(params.CarCfg.params.EOG_ch_names);
        do_car     = true;
        yaml_src   = 'YAML';
    catch
        fs         = double(py.freq);
        chunk_size = double(py.chunk_size);
        bufsize    = double(py.window_size);
        eog_names  = {};
        do_car     = true;
        yaml_src   = '.mat defaults';
        fprintf('  [WARN] No companion YAML — using .mat defaults (no EOG exclusion).\n');
    end
    fprintf('  fs=%.0f  chunk=%d  bufsize=%d  params=%s\n', fs, chunk_size, bufsize, yaml_src);

    proc_cfg = struct('samplerate', fs, 'chunk_size', chunk_size, ...
                      'bufsize', bufsize, 'filter_order', 4, ...
                      'do_car', do_car, 'eog_names', {eog_names});

    % ── Run MATLAB pipeline ───────────────────────────────────────────────────
    [feat_ml, header_ml, ~, feat_pre_ml] = apply_processing(signal, header, csp_struct, proc_cfg);
    n_valid_ml = sum(~any(isnan(feat_ml), 2));
    fprintf('  MATLAB: %d chunks (%d valid)\n', size(feat_ml, 1), n_valid_ml);

    % ── Align and extract log-features ───────────────────────────────────────
    cf_idx = find(header_ml.EVENT.TYP == CF_CODE);
    fprintf('  CF events (781) in MATLAB header: %d\n', numel(cf_idx));

    feat_ml_log = NaN(N_win, n_bands * n_comp);
    k_ml_arr    = NaN(N_win, 1);
    for i = 1:N_win
        t = t_arr(i);                               % 0-based trial index
        j = j_arr(i);                               % 0-based window within trial
        if t + 1 > numel(cf_idx), continue; end
        pos_k = header_ml.EVENT.POS(cf_idx(t + 1)); % 1-based chunk at CF onset
        k_ml  = pos_k + j + 1;                      % aligned MATLAB chunk
        if k_ml < 1 || k_ml > size(feat_ml, 1), continue; end
        f_raw = feat_ml(k_ml, :);
        if any(isnan(f_raw)), continue; end
        feat_ml_log(i, :) = log(max(f_raw, 1e-300));
        k_ml_arr(i)        = k_ml;
    end

    valid = ~any(isnan(feat_ml_log), 2) & ~any(isnan(X_py_flat), 2);
    N_cmp = sum(valid);
    fprintf('  Comparable windows: %d / %d\n', N_cmp, N_win);

    if N_cmp == 0
        fprintf('  [WARN] No valid windows — check GDF path or alignment.\n');
        continue;
    end

    % ── Per-file MAE (post-CSP) ───────────────────────────────────────────────
    diff_f  = X_py_flat(valid, :) - feat_ml_log(valid, :);
    mae_f   = mean(abs(diff_f(:)));
    max_f   = max(abs(diff_f(:)));
    status  = 'OK';
    if mae_f >= 1e-5, status = 'HIGH — check CSP matrices'; end
    fprintf('  Post-CSP  MAE = %.2e   max = %.2e   [%s]\n', mae_f, max_f, status);

    % ── Pre-CSP comparison (isolate signal vs CSP issue) ─────────────────────
    pre_ml_log = NaN(N_win, n_bands * n_sel_ch);
    for i = 1:N_win
        t   = t_arr(i);
        j   = j_arr(i);
        if t + 1 > numel(cf_idx), continue; end
        pos_k = header_ml.EVENT.POS(cf_idx(t + 1));
        k_ml  = pos_k + j + 1;
        if k_ml < 1 || k_ml > size(feat_pre_ml, 1), continue; end
        f_pre = feat_pre_ml(k_ml, :);
        if any(isnan(f_pre)), continue; end
        pre_ml_log(i, :) = log(max(f_pre, 1e-300));
    end
    % Python pre-CSP is raw power (no log) — take log for comparison
    X_pre_py_log = log(max(X_pre_py, 1e-300));
    pre_valid = ~any(isnan(pre_ml_log), 2) & ~any(isnan(X_pre_py_log), 2);
    diff_pre  = X_pre_py_log(pre_valid, :) - pre_ml_log(pre_valid, :);
    mae_pre   = mean(abs(diff_pre(:)));
    max_pre   = max(abs(diff_pre(:)));
    if mae_pre < 1e-5
        pre_status = 'OK — signals match, issue is in CSP matrices';
    elseif mae_pre < 0.5
        pre_status = 'SMALL — minor filter/CAR difference';
    else
        pre_status = 'LARGE — check signal loading/units or bandpass';
    end
    fprintf('  Pre-CSP   MAE = %.2e   max = %.2e   [%s]\n', mae_pre, max_pre, pre_status);

    all_py = [all_py; X_py_flat(valid, :)];    %#ok<AGROW>
    all_ml = [all_ml; feat_ml_log(valid, :)];  %#ok<AGROW>

    % Save first file's data for trace diagnostic
    if fi == 1
        feat_ml_log_raw = log(max(feat_ml, 1e-300));  % full trace (NaN where invalid)
        feat_ml_log_raw(any(isnan(feat_ml), 2), :) = NaN;
        trace1.feat_ml_all = feat_ml_log_raw;
        trace1.feat_ml_log = feat_ml_log;
        trace1.X_py_flat   = X_py_flat;
        trace1.k_ml_arr    = k_ml_arr;
        trace1.valid       = valid;
        trace1.n_bands     = n_bands;
        trace1.n_comp      = n_comp;
        trace1.bands       = py.bands;
    end
end

% ── Aggregate report ──────────────────────────────────────────────────────────
if isempty(all_py)
    fprintf('\nNo valid data — nothing to plot.\n');
    return;
end

diff_all = all_py - all_ml;
mae_all  = mean(abs(diff_all(:)));
max_all  = max(abs(diff_all(:)));
N_all    = size(all_py, 1);

fprintf('\n══ Aggregate (%d windows, %d file(s)) ════════════════════\n', N_all, n_files);
fprintf('  MAE = %.2e   max = %.2e\n', mae_all, max_all);
fprintf('  Expected MAE < 1e-5 (sosfilt vs ba-form numerical difference).\n');

% ── Per-feature MAE breakdown (band x component) ─────────────────────────────
mae_per  = mean(abs(diff_all), 1);                  % [1, n_bands*n_comp]
mae_grid = reshape(mae_per, n_comp, n_bands)';       % [n_bands, n_comp]
fprintf('\n  Per-feature MAE (rows=bands, cols=CSP comp):\n');
for b = 1:n_bands
    fprintf('    Band %d [%.1f-%.1f Hz]:', b, py.bands(b, 1), py.bands(b, 2));
    fprintf('  %8.2e', mae_grid(b, :));
    fprintf('\n');
end

% ── Plots ─────────────────────────────────────────────────────────────────────
figure('Name', 'Training features: Python vs MATLAB', 'Color', 'w', 'NumberTitle', 'off');

subplot(1, 2, 1); hold on; grid on;
plot(all_ml(:), all_py(:), '.', 'MarkerSize', 3, 'Color', [0.2 0.5 0.9]);
lo = min([all_ml(:); all_py(:)]);
hi = max([all_ml(:); all_py(:)]);
plot([lo hi], [lo hi], 'k--', 'HandleVisibility', 'off');
xlabel('MATLAB log-feat');
ylabel('Python log-feat');
title(sprintf('Feature scatter  (%d windows x %d feat)', N_all, size(all_py, 2)));

subplot(1, 2, 2); hold on; grid on;
err_per_win = max(abs(diff_all), [], 2);
plot(err_per_win, '-', 'Color', [0.2 0.5 0.9]);
set(gca, 'YScale', 'log');
xlabel('Window index (all files)');
ylabel('max|feat_{py} - feat_{ml}|');
title('Per-window max feature error');
yline(1e-5, 'r--', 'threshold 1e-5', 'HandleVisibility', 'off');

sgtitle(sprintf('validate\\_features.m  —  MAE=%.2e  (%d windows, %d file(s))', ...
        mae_all, N_all, n_files), 'Interpreter', 'none');

% ── Trace diagnostic: min / median / max MAE feature (first file) ────────────
if ~isempty(trace1.feat_ml_all)
    nc       = trace1.n_comp;
    chunk_ax = (1:size(trace1.feat_ml_all, 1))';
    k_valid  = trace1.k_ml_arr(trace1.valid);

    % Pick 3 features by MAE rank across all files
    [mae_sorted, sort_idx] = sort(mae_per);
    n_feat  = numel(mae_per);
    pick_idx = [sort_idx(1), ...
                sort_idx(round(n_feat / 2)), ...
                sort_idx(end)];
    pick_lbl = {'min MAE', 'median MAE', 'max MAE'};

    figure('Name', 'CSP feature traces: min / median / max MAE (file 1)', ...
           'Color', 'w', 'NumberTitle', 'off');

    for pi = 1:3
        fi_idx = pick_idx(pi);                   % feature column index (band-major)
        b_idx  = ceil(fi_idx / nc);              % 1-based band
        c_idx  = mod(fi_idx - 1, nc) + 1;        % 1-based component

        ax = subplot(3, 1, pi);
        hold on; grid on;

        % Full MATLAB log-feature trace
        plot(chunk_ax, trace1.feat_ml_all(:, fi_idx), '-', ...
             'Color', [0.75 0.75 0.75], 'LineWidth', 0.8);

        % Python values at k_ml positions
        scatter(k_valid, trace1.X_py_flat(trace1.valid, fi_idx), ...
                10, [0.85 0.33 0.1], 'filled');

        % MATLAB values at same k_ml positions
        scatter(k_valid, trace1.feat_ml_log(trace1.valid, fi_idx), ...
                10, [0.2 0.5 0.9], 'filled');

        band_lo = trace1.bands(b_idx, 1);
        band_hi = trace1.bands(b_idx, 2);
        ylabel('log-feat');
        title(sprintf('%s  —  Band %d [%.0f-%.0f Hz], Comp %d  (MAE=%.2e)', ...
              pick_lbl{pi}, b_idx, band_lo, band_hi, c_idx, mae_per(fi_idx)));
        set(ax, 'FontSize', 8);
    end

    legend({'MATLAB (all chunks)', 'Python (CF windows)', 'MATLAB (aligned)'}, ...
           'Orientation', 'horizontal', 'Location', 'southoutside', 'FontSize', 8);
    xlabel('MATLAB chunk index');
    sgtitle('log-feature traces — file 1', 'Interpreter', 'none');
end
