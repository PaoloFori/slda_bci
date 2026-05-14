#!/usr/bin/env python3
"""
sLDA Python reference — replicates slda.py ROS node output for validation.

Modes:
  'csv': reads pre-computed FBCSP features (raw mean power from processing_bci),
         applies log + sLDA sigmoid.  Simple and fast.
  'gdf': runs the full pipeline from a GDF file:
         CAR → LP→HP (causal, order 4) → ring buffer → CSP → mean(x²) → log → sLDA.

Workflow (CSV):
  1. roslaunch slda_bci test_node_slda.launch
  2. Ctrl+C  →  test/slda_output.csv  +  test/slda_output_first_seq.txt
  3. python3 test_slda.py  (with input_mode = 'csv')

Workflow (GDF):
  1. roslaunch slda_bci test_node_slda_gdf.launch \\
         gdf_file:=$(rospack find processing_bci)/test/prova32ch.gdf
  2. Ctrl+C  →  test/slda_gdf_output.csv  +  test/slda_gdf_output_first_seq.txt
  3. python3 test_slda.py  (with input_mode = 'gdf')
"""

import os
import sys
import numpy as np
import yaml
from scipy.signal import butter, sosfilt
from scipy.special import expit
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
input_mode = 'gdf'   # 'csv' | 'gdf'

data_dir  = './test_node_data/'
out_dir   = './test_node_data/slda_bci/'
slda_yaml = './src/slda_bci/models/slda_test.yaml'
car_yaml  = './src/rosneuro_filters_car/cfg/car.yaml'
csp_yaml  = './src/processing_bci/cfg/csp/mi/csp_mi_test.yaml'

if input_mode == 'csv':
    fbcsp_file = data_dir + 'processing_bci/fbcsp_processing.csv'
    ros_file   = out_dir  + 'slda_output.csv'
    framerate  = 20
    samplerate = 500
else:
    gdf_file   = data_dir + 'prova32ch.gdf'
    ros_file   = out_dir  + 'slda_gdf_output.csv'
    framerate  = 16
    samplerate = 512

# ──────────────────────────────────────────────────────────────────
# first_seq
# ──────────────────────────────────────────────────────────────────
first_seq_file = ros_file.replace('.csv', '_first_seq.txt')
first_seq = 0
if os.path.isfile(first_seq_file):
    with open(first_seq_file) as fh:
        first_seq = int(fh.read().strip())
    print(f'ROS first_seq = {first_seq}')
else:
    print('first_seq file not found — assuming first_seq = 0.')

# ──────────────────────────────────────────────────────────────────
# Load sLDA model
# ──────────────────────────────────────────────────────────────────
with open(slda_yaml) as fh:
    sp = yaml.safe_load(fh)['sLDACfg']['params']

weights      = np.array(sp['slda_weights'])                  # [1 × n_feat]
intercept    = np.array(sp['slda_intercept'])                # [1]
classes      = sp['classes']
bands_model  = sp['bands']
n_comp_model = len(sp['selected_components_indices'])
n_feat       = len(bands_model) * n_comp_model
print(f'sLDA: classes={classes}, {len(bands_model)} bands × {n_comp_model} comp = {n_feat} features')

# ──────────────────────────────────────────────────────────────────
# Compute reference features
# ──────────────────────────────────────────────────────────────────
if input_mode == 'csv':
    # Pre-computed raw mean power (not log-transformed) from processing_bci
    features = np.loadtxt(fbcsp_file, delimiter=',')   # [n_frames, n_feat]
    print(f'CSV: {features.shape[0]} frames × {features.shape[1]} features')

else:   # 'gdf' — full pipeline
    import mne
    mne.set_log_level('WARNING')

    with open(car_yaml) as fh:
        eog_ch_names = yaml.safe_load(fh)['CarCfg']['params']['EOG_ch_names']

    with open(csp_yaml) as fh:
        cp = yaml.safe_load(fh)['CspCfg']['params']
    csp_matrices = [np.array(m) for m in cp['csp_matrices']]   # [n_comp × n_sel_ch] each
    selected_ch  = cp['selected_channels']
    bands_csp    = cp['bands']
    n_bands      = len(bands_csp)

    raw = mne.io.read_raw_gdf(gdf_file, preload=True)
    sr  = int(raw.info['sfreq'])

    # Drop stim/trigger channels (same as MATLAB sload n_eeg restriction)
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
    raw.pick(eeg_picks)
    ch   = raw.ch_names
    data = raw.get_data()   # [n_eeg, n_samples]

    # Case-insensitive, whitespace-stripped match (GDF pads names to 16 chars)
    ch_norm = [c.strip().lower() for c in ch]
    def find_ch(name):
        n = name.strip().lower()
        if n not in ch_norm:
            raise ValueError(f'Channel "{name}" not found. Available: {[c.strip() for c in ch]}')
        return ch_norm.index(n)

    excl = [find_ch(c) for c in eog_ch_names]
    mask = np.ones(len(ch), dtype=bool)
    mask[excl] = False
    data -= data[mask].mean(axis=0)

    sel_idx = [find_ch(c) for c in selected_ch]
    chunk_size = sr // framerate
    buf_size   = sr   # 1-second ring buffer

    # Design causal LP→HP filters — order 4 each, matching Fbcsp.cpp
    sos_pairs = []
    for l_freq, h_freq in bands_csp:
        sos_lp = butter(4, h_freq, btype='lowpass',  fs=sr, output='sos')
        sos_hp = butter(4, l_freq, btype='highpass', fs=sr, output='sos')
        sos_pairs.append((sos_lp, sos_hp))

    n_ch = data.shape[0]
    # Zero initial conditions — matches MATLAB zeros() init and ROS filter startup
    zi_lp = [np.zeros((s.shape[0], n_ch, 2)) for s, _ in sos_pairs]
    zi_hp = [np.zeros((s.shape[0], n_ch, 2)) for _, s in sos_pairs]
    # Unpack sos_pairs for filter application
    sos_lp_list = [s[0] for s in sos_pairs]
    sos_hp_list = [s[1] for s in sos_pairs]

    # Ring buffers: NaN-initialized [buf_size × n_ch × n_bands]
    buffers = np.full((buf_size, n_ch, n_bands), np.nan)

    n_frames = data.shape[1] // chunk_size
    n_comp   = csp_matrices[0].shape[0]
    features_list = []

    for seq in range(n_frames):
        chunk = data[:, seq * chunk_size : (seq + 1) * chunk_size]   # [n_ch, chunk_size]

        for b in range(n_bands):
            chunk_lp, zi_lp[b] = sosfilt(sos_lp_list[b], chunk, zi=zi_lp[b])
            chunk_bp, zi_hp[b] = sosfilt(sos_hp_list[b], chunk_lp, zi=zi_hp[b])
            # Shift-register: drop oldest samples, append new ones (transpose: samples × ch)
            buffers[:, :, b] = np.vstack([buffers[chunk_size:, :, b], chunk_bp.T])

        if np.any(np.isnan(buffers)):
            # Buffer not yet full — ROS outputs ones, sLDA sees log(1)=0 → p=sigmoid(b)
            features_list.append(np.ones(n_bands * n_comp))
            continue

        frame_feats = []
        for b in range(n_bands):
            buf_sel = buffers[:, sel_idx, b]         # [buf_size × n_sel]
            csp_out = buf_sel @ csp_matrices[b].T    # [buf_size × n_comp]
            frame_feats.extend(np.mean(csp_out ** 2, axis=0))   # mean(x²) per component
        features_list.append(frame_feats)

    features = np.array(features_list)
    print(f'GDF: {features.shape[0]} frames × {features.shape[1]} features '
          f'(sr={sr}, chunkSize={chunk_size}, bufSize={buf_size})')

# Apply log + sLDA (mirrors slda.py: log → w·x+b → sigmoid)
log_feat  = np.log(features)
scores    = (log_feat @ weights.T + intercept).ravel()
probs_c2  = expit(scores)
probs_c1  = 1.0 - probs_c2
py_output = np.column_stack([probs_c1, probs_c2])   # [n_frames, 2]
print(f'Python output: {py_output.shape[0]} frames')

# ──────────────────────────────────────────────────────────────────
# Compare with ROS output
# ──────────────────────────────────────────────────────────────────
if not os.path.isfile(ros_file):
    print(f'ROS output not found: {ros_file}  (run the launch file first)')
    sys.exit(0)

ros_data = np.loadtxt(ros_file, delimiter=',')   # [n_ros_frames, 2]
print(f'ROS output: {ros_data.shape[0]} frames')

n_cmp = min(len(ros_data), len(py_output))
ros_out = ros_data[:n_cmp]
mat_out = py_output[:n_cmp]

# Skip frames before first_seq
skip = min(first_seq, n_cmp)
ros_cmp = ros_out[skip:]
mat_cmp = mat_out[skip:]

# Cross-correlation on p(class 2) to detect residual lag
MAX_LAG = 20   # frames
ch_ref  = 1    # class-2 column
n_xcorr = min(len(ros_cmp), len(mat_cmp))
r_ref = ros_cmp[:n_xcorr, ch_ref] - ros_cmp[:n_xcorr, ch_ref].mean()
m_ref = mat_cmp[:n_xcorr, ch_ref] - mat_cmp[:n_xcorr, ch_ref].mean()
[xcf, lags] = np.array(
    [(np.correlate(r_ref, np.roll(m_ref, k))[0], k)
     for k in range(-MAX_LAG, MAX_LAG + 1)]
).T
lag = int(lags[np.argmax(xcf)])
print(f'Cross-corr lag: {lag:+d} frame(s)  ', end='')
print('[no residual lag]' if lag == 0 else '[ROS lags Python]' if lag > 0 else '[Python lags ROS]')

if lag > 0:
    r_al = ros_cmp[lag:]; m_al = mat_cmp[:len(ros_cmp)-lag]
elif lag < 0:
    r_al = ros_cmp[:len(mat_cmp)+lag]; m_al = mat_cmp[-lag:]
else:
    r_al = ros_cmp; m_al = mat_cmp

n_al = min(len(r_al), len(m_al))
r_al, m_al = r_al[:n_al], m_al[:n_al]

fr = samplerate // (samplerate // framerate)   # = framerate
t_raw = np.arange(n_xcorr) / fr
t_al  = np.arange(n_al) / fr

mae_raw     = np.mean(np.abs(r_ref - m_ref))
mae_aligned = np.mean(np.abs(r_al[:, ch_ref] - m_al[:, ch_ref]))
print(f'MAE p(c2) [raw]     : {mae_raw:.6f}')
print(f'MAE p(c2) [aligned] : {mae_aligned:.6f}')

# ──────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].plot(t_raw, ros_cmp[:n_xcorr, ch_ref], 'b',    lw=1.5, label='ROS node')
ax[0].plot(t_raw, mat_cmp[:n_xcorr, ch_ref], 'r--',  lw=1,   label='Python ref')
ax[0].set_title(f'[RAW] sLDA | p(c2) | mode={input_mode} | first_seq={first_seq}')
ax[0].set_ylabel('probability'); ax[0].legend(); ax[0].grid()
ax[1].bar(t_raw, np.abs(ros_cmp[:n_xcorr, ch_ref] - mat_cmp[:n_xcorr, ch_ref]))
ax[1].set_xlabel('time [s]'); ax[1].set_ylabel('|diff|')
ax[1].set_title(f'Differences (lag={lag:+d} frames)'); ax[1].grid()
plt.tight_layout()

fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6))
ax2[0].plot(t_al, r_al[:, ch_ref], 'b',   lw=1.5, label='ROS node')
ax2[0].plot(t_al, m_al[:, ch_ref], 'r--', lw=1,   label='Python ref')
ax2[0].set_title(f'[ALIGNED lag={lag:+d}] sLDA | p(c2)')
ax2[0].set_ylabel('probability'); ax2[0].legend(); ax2[0].grid()
ax2[1].bar(t_al, np.abs(r_al[:, ch_ref] - m_al[:, ch_ref]))
ax2[1].set_xlabel('time [s]'); ax2[1].set_ylabel('|diff|')
ax2[1].set_title('Differences after alignment'); ax2[1].grid()
plt.tight_layout()

plt.show()
