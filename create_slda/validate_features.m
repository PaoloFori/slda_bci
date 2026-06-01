%% VALIDATE_FEATURES  Stage-by-stage comparison of Python (create_slda) vs MATLAB
%   (apply_processing) feature extraction on the same calibration GDF(s).
%
%   Compares three stages:
%     Stage 1 — pre-CSP  : per-channel mean power (ring buffer output, before spatial filter)
%     Stage 2 — post-CSP : per-component mean power (CSP applied, no log)
%     Stage 3 — pre-sLDA : log of post-CSP power (= sLDA input)
%
%   Root causes of mismatch (in order of impact):
%     1. CAR channel count: BIOSIG may load more channels than MNE's N_CHANNELS.
%        Saved n_channels in the .mat is used to strip MATLAB's signal before CAR.
%     2. ba-form vs sosfilt: MATLAB butter+filter vs Python sosfilt can differ
%        by ~1e-4 for low-frequency high-pass filters (cutoffs < 15 Hz / Nyquist).
%
%   Expected MAE with fix applied:
%     Stage 1 < 1e-3  (residual from ba-form vs sosfilt at low HP cutoffs)
%     Stage 3 < 1e-3  (same root cause, amplified by log for small values)

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
if isequal(mat_names, 0), error('validate_features:cancel', 'No file selected.'); end
if ischar(mat_names), mat_names = {mat_names}; end

CF_CODE = 781;
n_files = numel(mat_names);

% Accumulators for aggregate MAE (stage 3 = sLDA input)
agg_py = [];  agg_ml = [];

% First-file trace data for figures
trace1 = struct('valid', [], 'valid1', [], 'k_ml_arr', [], ...
                'py_log', [], 'ml_log', [], ...
                'py_out', [], 'ml_out', [], ...
                'py_pre', [], 'ml_pre', [], ...
                't_arr',  [], 'j_arr',  [], ...
                'sel_feat_idx', [], 'sel_ch_names', {{}}, ...
                'n_bands', 0, 'n_comp', 0, 'bands', []);

for fi = 1:n_files
    mat_path = fullfile(mat_dir, mat_names{fi});
    fprintf('\n[%d/%d] %s\n', fi, n_files, mat_names{fi});

    py      = load(mat_path);
    n_bands = round(py.n_bands);
    n_comp  = round(py.n_csp_comp);
    j_arr   = round(py.j_in_trial(:));
    t_arr   = round(py.trial_idx(:));
    N_win   = numel(j_arr);

    % ── selected_feature_indices (0-based band-major Python layout) ───────────
    if isfield(py, 'selected_feature_indices')
        sel_feat_idx = round(py.selected_feature_indices(:))';
    else
        sel_feat_idx = 0 : (n_comp * n_bands - 1);
    end

    % ── Reshape Python arrays ─────────────────────────────────────────────────
    % X_csp_log / X_csp_out : [N_win, N_BANDS, N_COMP]
    %   permute→[N_win,N_COMP,N_BANDS], reshape col-major → comp-fast band-slow
    %   matches apply_processing column-major flatten of [n_comp, n_bands]
    X_log_flat = reshape(permute(py.X_csp_log, [1,3,2]), N_win, n_comp*n_bands);
    X_out_flat = reshape(permute(py.X_csp_out, [1,3,2]), N_win, n_comp*n_bands);

    % X_pre_csp : [N_win, N_BANDS, N_SEL_CH]
    %   permute→[N_win,N_SEL_CH,N_BANDS], reshape col-major → ch-fast band-slow
    %   matches features_pre from apply_processing (band-major: [b0_ch0..b0_chN, b1_ch0..])
    n_sel_ch   = size(py.X_pre_csp, 3);
    X_pre_flat = reshape(permute(py.X_pre_csp, [1,3,2]), N_win, n_sel_ch*n_bands);

    fprintf('  Python: %d windows | %d bands x %d comp | %d sel channels\n', ...
            N_win, n_bands, n_comp, n_sel_ch);

    % ── Build CSP struct ──────────────────────────────────────────────────────
    csp.n_bands           = n_bands;
    csp.n_components      = n_comp;
    csp.bands             = py.bands;
    csp.selected_channels = cellstr(py.selected_channels);
    csp.csp_matrices      = cell(n_bands, 1);
    for b = 1:n_bands
        csp.csp_matrices{b} = squeeze(py.csp_matrices(b, :, :));
    end

    % ── Load GDF ─────────────────────────────────────────────────────────────
    gdf_path = char(py.gdf_path);
    if ~isfile(gdf_path)
        fprintf('  [WARN] GDF not found: %s — skipping.\n', gdf_path);
        continue;
    end
    [signal, header, ~] = load_gdf(gdf_path);

    % ── FIX: strip to N_CHANNELS to match Python's MNE channel pick ──────────
    % BIOSIG may load extra channels (Status, ACC, etc.) beyond Python's N_CHANNELS.
    % Including them in the CAR average corrupts the reference signal.
    if isfield(py, 'n_channels')
        n_ch_py = round(py.n_channels);
        if size(signal, 2) > n_ch_py
            fprintf('  [FIX] BIOSIG loaded %d ch; stripping to first %d (= Python N_CHANNELS)\n', ...
                    size(signal, 2), n_ch_py);
            signal       = signal(:, 1:n_ch_py);
            header.Label = header.Label(1:n_ch_py);
        end
    end

    % ── Load processing params ────────────────────────────────────────────────
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
        if isfield(py, 'exclude_channels')
            eog_names = cellstr(py.exclude_channels);
        else
            eog_names = {};
        end
        do_car     = true;
        yaml_src   = '.mat defaults';
        if isempty(eog_names)
            fprintf('  [WARN] No companion YAML — using .mat defaults (no EOG exclusion).\n');
        else
            fprintf('  [WARN] No companion YAML — using .mat defaults (EOG excl from .mat: %s).\n', ...
                    strjoin(eog_names, ', '));
        end
    end
    fprintf('  CAR eog_names: [%s]\n', strjoin(eog_names, ', '));
    fprintf('  fs=%.0f  chunk=%d  bufsize=%d  params=%s  signal=%dx%d\n', ...
            fs, chunk_size, bufsize, yaml_src, size(signal,1), size(signal,2));

    % ── Signal RMS sanity check ───────────────────────────────────────────────
    rms_ml = sqrt(mean(signal(:).^2, 'omitnan'));
    fprintf('  MATLAB signal RMS = %.3g  (expect ~10-100 if µV, ~1e-5 if V)\n', rms_ml);

    proc_cfg = struct('samplerate', fs, 'chunk_size', chunk_size, ...
                      'bufsize', bufsize, 'filter_order', 4, ...
                      'do_car', do_car, 'eog_names', {eog_names});

    % ── Run MATLAB pipeline ───────────────────────────────────────────────────
    [feat_ml, header_ml, ~, feat_pre_ml] = apply_processing(signal, header, csp, proc_cfg);
    n_chunks = size(feat_ml, 1);
    fprintf('  MATLAB: %d chunks (%d valid)\n', n_chunks, sum(~any(isnan(feat_ml),2)));

    cf_idx = find(header_ml.EVENT.TYP == CF_CODE);
    fprintf('  CF events (781): %d\n', numel(cf_idx));

    % ── Align per-window features ─────────────────────────────────────────────
    ml_log = NaN(N_win, n_comp*n_bands);
    ml_out = NaN(N_win, n_comp*n_bands);
    ml_pre = NaN(N_win, n_sel_ch*n_bands);
    k_ml_arr = NaN(N_win, 1);

    for i = 1:N_win
        t = t_arr(i);  j = j_arr(i);
        if t + 1 > numel(cf_idx), continue; end
        pos_k = header_ml.EVENT.POS(cf_idx(t + 1));
        dur_k = header_ml.EVENT.DUR(cf_idx(t + 1));
        k_ml  = pos_k + j + 1;
        if k_ml < 1 || k_ml > n_chunks || j + 1 > dur_k, continue; end
        f3 = feat_ml(k_ml, :);
        f2 = feat_pre_ml(k_ml, :);
        if any(isnan(f3)) || any(isnan(f2)), continue; end
        ml_log(i,:)  = log(max(f3, 1e-300));
        ml_out(i,:)  = f3;
        ml_pre(i,:)  = f2;
        k_ml_arr(i)  = k_ml;
    end

    valid3 = ~any(isnan(ml_log),  2) & ~any(isnan(X_log_flat), 2);
    valid2 = ~any(isnan(ml_out),  2) & ~any(isnan(X_out_flat), 2);
    valid1 = ~any(isnan(ml_pre),  2) & ~any(isnan(X_pre_flat), 2);
    N_cmp  = sum(valid3);
    fprintf('  Aligned windows: %d / %d\n', N_cmp, N_win);
    if N_cmp == 0
        fprintf('  [WARN] No aligned windows — check GDF path or alignment.\n');
        continue;
    end

    % ── MAE ───────────────────────────────────────────────────────────────────
    diff1 = X_pre_flat(valid1,:) - ml_pre(valid1,:);
    mae1  = mean(abs(diff1(:)));  max1 = max(abs(diff1(:)));

    diff2 = X_out_flat(valid2,:) - ml_out(valid2,:);
    mae2  = mean(abs(diff2(:)));  max2 = max(abs(diff2(:)));

    diff3 = X_log_flat(valid3,:) - ml_log(valid3,:);
    mae3  = mean(abs(diff3(:)));  max3 = max(abs(diff3(:)));

    s1 = 'OK'; s3 = 'OK';
    if mae1 >= 1e-5, s1 = 'diff (check CAR channels or filter form)'; end
    if mae3 >= 1e-3, s3 = 'HIGH';
        if mae1 < 1e-5, s3 = [s3 ' — signals match but CSP matrices differ']; end
    end

    fprintf('  Stage 1  pre-CSP      MAE=%.2e  max=%.2e  [%s]\n', mae1, max1, s1);
    fprintf('  Stage 2  post-CSP     MAE=%.2e  max=%.2e\n',        mae2, max2);
    fprintf('  Stage 3  pre-sLDA     MAE=%.2e  max=%.2e  [%s]\n', mae3, max3, s3);

    agg_py = [agg_py; X_log_flat(valid3,:)];  %#ok<AGROW>
    agg_ml = [agg_ml; ml_log(valid3,:)];       %#ok<AGROW>

    if fi == 1
        trace1.py_log       = X_log_flat;  trace1.ml_log  = ml_log;
        trace1.py_out       = X_out_flat;  trace1.ml_out  = ml_out;
        trace1.py_pre       = X_pre_flat;  trace1.ml_pre  = ml_pre;
        trace1.k_ml_arr     = k_ml_arr;
        trace1.t_arr        = t_arr;
        trace1.j_arr        = j_arr;
        trace1.valid        = valid3;
        trace1.valid1       = valid1;
        trace1.n_bands      = n_bands;
        trace1.n_comp       = n_comp;
        trace1.bands        = py.bands;
        trace1.sel_feat_idx = sel_feat_idx;
        trace1.sel_ch_names = csp.selected_channels;
    end
end

% ── Aggregate report ──────────────────────────────────────────────────────────
if isempty(agg_py)
    fprintf('\nNo valid data — nothing to plot.\n'); return;
end
diff_all = agg_py - agg_ml;
mae_all  = mean(abs(diff_all(:)));
max_all  = max(abs(diff_all(:)));
N_all    = size(agg_py, 1);

fprintf('\n══ Aggregate stage 3 (%d windows, %d file(s)) ════════════════\n', N_all, n_files);
fprintf('  MAE=%.2e   max=%.2e\n', mae_all, max_all);
mae_per  = mean(abs(diff_all), 1);
n_comp_r = trace1.n_comp;  n_bands_r = trace1.n_bands;
mae_grid = reshape(mae_per, n_comp_r, n_bands_r)';
fprintf('\n  Per-feature MAE (rows=bands, cols=CSP comp):\n');
py_last = load(fullfile(mat_dir, mat_names{end}));
for b = 1:n_bands_r
    fprintf('    Band %d [%.1f-%.1f Hz]:', b, py_last.bands(b,1), py_last.bands(b,2));
    fprintf('  %8.2e', mae_grid(b,:));
    fprintf('\n');
end



% ═════════════════════════════════════════════════════════════════════════════
% FIG 1 — Pre-CSP: power for ONE trial × ONE channel across all bands
%   Shows the per-frame power (ring buffer mean(x²)) as it evolves during
%   the CF window, for the first selected channel. Isolates filter differences
%   from alignment issues: if the two traces track each other with a small
%   offset the filter form differs; if they are shifted in time the alignment
%   formula is wrong.
% ═════════════════════════════════════════════════════════════════════════════
if ~isempty(trace1.valid1) && any(trace1.valid1)
    nb_t  = trace1.n_bands;
    nsc_t = round(size(trace1.ml_pre, 2) / nb_t);

    % Pick the trial with the most valid windows (most informative)
    t_unique = unique(trace1.t_arr(trace1.valid1));
    n_per_t  = arrayfun(@(t) sum(trace1.t_arr == t & trace1.valid1), t_unique);
    [~, best] = max(n_per_t);
    t_pick = t_unique(best);

    trial_mask = (trace1.t_arr == t_pick) & trace1.valid1;
    j_trial    = trace1.j_arr(trial_mask);          % 0-based window-within-trial
    [~, ord]   = sort(j_trial);
    trial_idx  = find(trial_mask);
    trial_idx  = trial_idx(ord);                    % sorted by j
    w_ax       = trace1.j_arr(trial_idx);           % x: window index within trial

    % Channel index 0 = first selected channel
    ch_pick = 1;   % 1-based into selected_channels
    ch_name = '';
    if ~isempty(trace1.sel_ch_names) && numel(trace1.sel_ch_names) >= ch_pick
        ch_name = trace1.sel_ch_names{ch_pick};
    end

    figure('Name','Fig 1 — Pre-CSP: one trial, one channel', 'Color','w', 'NumberTitle','off');
    for b = 1:nb_t
        col_ch = (b-1)*nsc_t + ch_pick;   % band-major column for (band b, channel ch_pick)
        py_b   = trace1.py_pre(trial_idx, col_ch);
        ml_b   = trace1.ml_pre(trial_idx, col_ch);
        mae_b  = mean(abs(py_b - ml_b));
        ltitle = sprintf('Band %d [%.0f-%.0f Hz]  ch=%s', ...
                         b, trace1.bands(b,1), trace1.bands(b,2), ch_name);
        rtitle = sprintf('|Δ|  MAE=%.2e', mae_b);
        plot_comparison_row(nb_t, 2, b, w_ax, py_b, ml_b, ltitle, rtitle, b==1);
    end
    subplot(nb_t, 2, (nb_t-1)*2+1); xlabel('window index within trial');
    subplot(nb_t, 2, (nb_t-1)*2+2); xlabel('window index within trial');
    sgtitle(sprintf('Pre-CSP: ring-buffer power  trial=%d  ch=%s  (Python vs MATLAB, file 1)', ...
                    t_pick, ch_name), 'Interpreter','none');
end

% Helper: re-extract the same trial used in Fig 1 for Figs 2 and 3
if ~isempty(trace1.valid) && any(trace1.valid)
    t_unique3  = unique(trace1.t_arr(trace1.valid));
    n_per_t3   = arrayfun(@(t) sum(trace1.t_arr == t & trace1.valid), t_unique3);
    [~, best3] = max(n_per_t3);
    t_pick3    = t_unique3(best3);

    trial_mask3 = (trace1.t_arr == t_pick3) & trace1.valid;
    j_trial3    = trace1.j_arr(trial_mask3);
    [~, ord3]   = sort(j_trial3);
    trial_idx3  = find(trial_mask3);
    trial_idx3  = trial_idx3(ord3);
    w_ax3       = trace1.j_arr(trial_idx3);
end

% ═════════════════════════════════════════════════════════════════════════════
% FIG 2 — Post-CSP (linear): sLDA-selected features, same trial as Fig 1
% ═════════════════════════════════════════════════════════════════════════════
if ~isempty(trace1.valid) && any(trace1.valid)
    sel = trace1.sel_feat_idx;
    ns  = min(8, numel(sel));

    figure('Name','Fig 2 — Post-CSP linear (sLDA selected)', 'Color','w', 'NumberTitle','off');
    for ki = 1:ns
        mi  = sel(ki) + 1;
        b_i = ceil(mi / trace1.n_comp);
        c_i = mod(mi - 1, trace1.n_comp) + 1;
        py_f  = trace1.py_out(trial_idx3, mi);
        ml_f  = trace1.ml_out(trial_idx3, mi);
        mae_f = mean(abs(py_f - ml_f));
        ltitle = sprintf('Band %d [%.0f-%.0f Hz]  Comp %d', ...
                         b_i, trace1.bands(b_i,1), trace1.bands(b_i,2), c_i);
        rtitle = sprintf('|Δ|  MAE=%.2e', mae_f);
        plot_comparison_row(ns, 2, ki, w_ax3, py_f, ml_f, ltitle, rtitle, ki==1);
    end
    subplot(ns, 2, (ns-1)*2+1); xlabel('window index within trial');
    subplot(ns, 2, (ns-1)*2+2); xlabel('window index within trial');
    sgtitle(sprintf('Post-CSP power (linear) — sLDA selected features  trial=%d  (Python vs MATLAB, file 1)', ...
                    t_pick3), 'Interpreter','none');
end

% ═════════════════════════════════════════════════════════════════════════════
% FIG 3 — Pre-sLDA (post-CSP + log): sLDA-selected features, same trial
% ═════════════════════════════════════════════════════════════════════════════
if ~isempty(trace1.valid) && any(trace1.valid)
    sel = trace1.sel_feat_idx;
    ns  = min(8, numel(sel));

    figure('Name','Fig 3 — Pre-sLDA log features (sLDA selected)', 'Color','w', 'NumberTitle','off');
    for ki = 1:ns
        mi  = sel(ki) + 1;
        b_i = ceil(mi / trace1.n_comp);
        c_i = mod(mi - 1, trace1.n_comp) + 1;
        py_f  = trace1.py_log(trial_idx3, mi);
        ml_f  = trace1.ml_log(trial_idx3, mi);
        mae_f = mean(abs(py_f - ml_f));
        ltitle = sprintf('Band %d [%.0f-%.0f Hz]  Comp %d  (log)', ...
                         b_i, trace1.bands(b_i,1), trace1.bands(b_i,2), c_i);
        rtitle = sprintf('|Δ|  MAE=%.2e', mae_f);
        plot_comparison_row(ns, 2, ki, w_ax3, py_f, ml_f, ltitle, rtitle, ki==1);
    end
    subplot(ns, 2, (ns-1)*2+1); xlabel('window index within trial');
    subplot(ns, 2, (ns-1)*2+2); xlabel('window index within trial');
    sgtitle(sprintf('Pre-sLDA: log(post-CSP power) — sLDA selected features  trial=%d  (Python vs MATLAB, file 1)', ...
                    t_pick3), 'Interpreter','none');
end

% ═════════════════════════════════════════════════════════════════════════════
% Helper: plot one comparison row (left=overlaid, right=|diff| with max)
% ═════════════════════════════════════════════════════════════════════════════
function plot_comparison_row(fig_rows, fig_cols, row_idx, w_ax, py_vals, ml_vals, ...
                              left_title, right_title, show_legend)
    dif = abs(py_vals - ml_vals);
    [mx_val, mx_i] = max(dif);

    subplot(fig_rows, 2, (row_idx-1)*2 + 1); hold on; grid on;
    plot(w_ax, py_vals, '-',  'Color',[0.85 0.33 0.1], 'LineWidth',1.0);
    plot(w_ax, ml_vals, '--', 'Color',[0.2  0.5  0.9], 'LineWidth',1.0);
    ylabel('value');
    title(left_title, 'FontSize',8, 'Interpreter','none');
    if show_legend
        legend({'Python','MATLAB'}, 'Location','northwest', 'FontSize',7);
    end

    subplot(fig_rows, 2, (row_idx-1)*2 + 2); hold on; grid on;
    plot(w_ax, dif, '-', 'Color',[0.5 0.0 0.5], 'LineWidth',1.0);
    plot(double(mx_i), mx_val, 'rv', 'MarkerSize',7, 'MarkerFaceColor','r');
    text(double(mx_i), mx_val, sprintf('  max=%.2e', mx_val), 'FontSize',7.5, 'Color','r');
    ylabel('|diff|');
    title(right_title, 'FontSize',8, 'Interpreter','none');
end