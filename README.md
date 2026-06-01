# slda_bci

ROS node implementing a **Shrinkage Linear Discriminant Analysis (sLDA)** classifier for online BCI decoding. It subscribes to FBCSP features published by `processing_bci`, classifies each frame, and publishes class probabilities as a `NeuroOutput` message.

---

## 1. Algorithm

For each incoming `eeg_fbcsp` frame:

1. **Feature extraction**: read `data[]` from the message (raw mean power `mean(x²)` per CSP component per band), reorder to match the model's expected band order.
2. **Log transform**: `log(mean(x²))` — converts power to log-power, matching the training convention.
3. **Feature selection mask** (optional): if the YAML carries `selected_feature_indices`, keep only those entries of the log-feature vector. Indices are into the *band-major flat vector* `[band0_comp0, band0_comp1, ..., band1_comp0, ...]`: for index `i`, `band = i // n_selected_components`, `component = selected_components_indices[i % n_selected_components]`. The loaded `slda_weights` are sized to match (one weight per kept index). Missing/empty field → no masking (backward-compatible).
4. **Linear discriminant**: `score = w · x + b` where `w` is the LDA weight vector and `b` is the intercept, both loaded from the YAML model.
5. **Platt Calibration**: Apply unbiased Platt scaling to the raw decision score using parameters `platt_a` and `platt_b` loaded from the model:
   $$P(c_2) = \frac{1}{1 + e^{-(\text{platt\_a} \cdot \text{score} + \text{platt\_b})}}$$
   $$P(c_1) = 1 - P(c_2)$$
   If Platt parameters are not present in the YAML, they default to standard sigmoid scaling (`platt_a = 1.0`, `platt_b = 0.0`).
6. **Publish** `/{paradigm}/neuroprediction/raw` (`rosneuro_msgs/NeuroOutput`) with `softpredict` = `[p(c₁), p(c₂)]`.

The classifier is reconstructed at startup from `coef_` and `intercept_` stored in the YAML, using `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr')`. Startup validation in `slda.py` checks that `coef_.shape[1] == len(selected_feature_indices)` (or `== nfeatures` if no mask), and that all selected indices fall within `[0, nfeatures)`.

---

## 2. Parameters

| ROS param | Type | Default | Description |
|-----------|------|---------|-------------|
| `~path_slda_model` | string | **required** | Absolute path to the sLDA YAML model file |
| `~topic_sub` | string | `/eeg_fbcsp` | Topic to subscribe for FBCSP features |
| `~paradigm` | string | **required** | Paradigm name (`mi` or `cvsa`); sets the output topic `/{paradigm}/neuroprediction/raw` |

### YAML model file structure

```yaml
sLDACfg:
  params:
    subject: S01
    classes: ['769', '770']           # GDF event codes for the two classes
    bands:                            # frequency bands used during training
      - [8, 12]
      - [12, 16]
      - [16, 20]
      - [20, 26]
    selected_channels: [FC5, FC1, C3, CP5, CP1, CP6, CP2, Cz, C4, FC6, FC2]
    selected_components_indices: [0, 1, 2, 3]
    # Optional band-major feature mask (see header comment auto-generated in the file).
    # Index i maps to (band = i // n_sel_comp, component = selected_components_indices[i % n_sel_comp]).
    selected_feature_indices: [0, 2, 5, 8, 11, 14]      # length must equal slda_weights[0] length
    feature_selection_method: mrmr                      # 'fisher' | 'mibif' | 'mrmr' | absent → no mask
    cv_mean_acc: 0.93
    cv_std_acc:  0.02
    slda_weights:   [[w0, w1, ..., wK]]                # [1 × K], K = len(selected_feature_indices) or n_features
    slda_intercept: [b]                                # scalar
    slda_calibrated_weights: [platt_a]                 # Platt scaling weight coefficient
    slda_calibrated_intercept: [platt_b]               # Platt scaling bias intercept
    csp: /path/to/csp_model.yaml
```

The node reads `classes`, `bands`, `selected_components_indices`, `slda_weights`, `slda_intercept`, optional `selected_feature_indices` + `feature_selection_method`, and optional `slda_calibrated_weights` / `slda_calibrated_intercept`. The other fields are metadata saved for traceability.

The notebook prepends a human-readable header block (`# rank | band | CSP component`) directly above `selected_feature_indices` so you can inspect which features survived without recomputing the mapping.

---

## 3. Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `~topic_sub` (e.g. `/mi/eeg_fbcsp`) | `processing_bci/eeg_fbcsp` | Subscribed | FBCSP features from `processing_bci` |
| `/{paradigm}/neuroprediction/raw` | `rosneuro_msgs/NeuroOutput` | Published | Softmax probabilities per frame |

### `eeg_fbcsp` message layout

`data[]` is stored column-major from a `[n_components × n_bands]` Eigen matrix:

```
[comp0_band0, comp1_band0, ..., compN_band0,   comp0_band1, ...]
```

`bands[]` is a flat `float32` array: `[l0, h0, l1, h1, ...]` (2 floats per band).

---

## 4. Usage

### 4a. Standalone (with pre-computed FBCSP features)

```bash
roslaunch slda_bci slda.launch \
    path_slda_model:=$(find slda_bci)/models/mi/slda_S01.yaml \
    paradigm:=mi
```

### 4b. Inside the full BCI pipeline

`slda.py` is launched by `launchers_bci/evaluation.launch`. Each paradigm (MI and CVSA) gets its own node instance with its own model:

```
/{paradigm}/eeg_fbcsp  (processing_bci)
    → slda_node  (slda.py)
        → /{paradigm}/neuroprediction/raw  (NeuroOutput)
```

---

## 5. Model Training (`create_slda/create_slda.ipynb`)

The notebook trains CSP + sLDA from calibration GDF files and saves models consumable by this node.

### Pipeline (matches online ROS processing exactly)

1. Load GDF → restrict to EEG channels (drop Status/trigger)
2. **CAR**: subtract per-sample mean of non-EOG channels (`EOG_ch_names = ['Fp1', 'Fp2']`)
3. **Bandpass** per band: causal Butterworth order 4, applied as **LP then HP** sequentially using **`scipy.signal.lfilter` with ba-form coefficients** (not `sosfilt` / SOS form). This matches `Fbcsp.cpp` and `apply_processing.m` exactly. Using `sosfilt` produces different transient behaviour for low normalised cutoffs (e.g. 8 Hz HP at 250 Hz Nyquist = 0.032) and would not match the online pipeline.
4. Extract 1-second sliding windows (step = `CHUNK_SIZE`) during continuous feedback (event 781)
   - **Trim start alignment**: the first training window starts at `trim_start_s = -0.95 s` relative to event 781 (CF onset), containing exactly 475 samples of cue and 25 samples of feedback. This replicates the exact initial ring buffer transient that the online classifier sees at runtime.
5. **CSP** (`n_components=4`, `reg='ledoit_wolf'`, `log=True`) on `SELECTED_CHANNELS` subset per band
6. **Feature computation**: `log(mean(x²))` per (component, band), concatenated in band-major order to a flat vector of length `n_bands × n_selected_components`
7. **Feature selection** (optional, controlled by `FS_METHOD`):
   - Methods: `'fisher'` (per-feature Fisher score), `'mibif'` (mutual information, Ang et al. 2012), `'mrmr'` (min-Redundancy Max-Relevance, Peng et al. 2005), or `'none'`
   - **Adaptive cap**: select up to `FS_K_TOP_MAX` features; the cap drops below the maximum when remaining scores fall below `FS_SCORE_FRAC × top_score` — avoids padding the set with low-discriminance features
   - **Voting CV aggregation**: each fold independently selects its own subset; features picked in ≥ ⌈K_FOLDS/2⌉ folds form the *stable set*, ordered by `(pick_count desc, mean_relevance desc)` and capped at `FS_K_TOP_MAX`. Falls back to top-K by mean relevance if too few features clear majority
   - The notebook prints per-fold `"FS kept N feat: [...]"` and a final table with rank/idx/band/component/folds-picked/mean-relevance
8. **Cross-validate** `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])` with `StratifiedGroupKFold(n_splits=K_FOLDS, groups=trials)` — prevents leakage from overlapping windows. Each fold trains on its *own* selected subset; the final model uses the voted stable set
9. **Platt Calibration Fitting**: Fit an unbiased 1D Logistic Regression on the out-of-fold CV test scores (`all_scores`) to calculate unbiased calibration coefficients (`platt_a` and `platt_b`) and avoid overfitting.
10. **Fold Sigma Monitoring**: A critical check is run on the cross-validation fold variance. If the fold standard deviation exceeds 8% (`std_acc > 0.08`), the notebook prints a warning recommending to reduce `N_CSP_COMPONENTS` to `2` to stabilize feature learning and halve feature dimensionality.
11. Train final model on all data using the stable feature set, save CSP yaml + sLDA yaml.
12. **Save figures** to `<gdf_dir>/images/<subject>_<paradigm>_<name>.png` — 13 PNG files covering: grid-search heatmap, CV and all-data ROC/calibration/confusion, training window density, per-CSP-component variance, LDA weights, CSP activation topomaps, scalp projections, feature heatmap, ERD/ERS topomaps, channel contribution bar chart.
13. **Save `_training_features.mat`** alongside each GDF — used by `validate_features.m` (see §5a). Saved fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `X_pre_csp` | `[N_win, N_BANDS, N_SEL_CH]` | Mean power per channel before CSP (ring buffer output) |
| `X_csp_out` | `[N_win, N_BANDS, N_COMP]` | Mean power per CSP component, no log |
| `X_csp_log` | `[N_win, N_BANDS, N_COMP]` | `log(mean power)` = sLDA input |
| `csp_matrices` | `[N_BANDS, N_COMP, N_SEL_CH]` | CSP filter matrices |
| `selected_channels` | `[N_SEL_CH]` | Channel names used for CSP |
| `selected_feature_indices` | `[K]` | 0-based band-major indices of sLDA-kept features |
| `bands` | `[N_BANDS, 2]` | `[lo, hi]` Hz per band |
| `j_in_trial` | `[N_win]` | 0-based window index within trial |
| `trial_idx` | `[N_win]` | 0-based trial index in file |
| `n_channels` | scalar | Python's `N_CHANNELS` — used by `validate_features.m` to strip BIOSIG's extra channels so the CAR reference matches |
| `exclude_channels` | `[N_EXCL]` | Python's `EXCLUDE_CHANNELS` (`['Fp1','Fp2']`) — fallback for `validate_features.m` when the companion YAML has no `CarCfg` (e.g. calibration recordings) |
| `freq`, `chunk_size`, `window_size` | scalar | Signal parameters |
| `n_csp_comp`, `n_bands` | scalar | Shape metadata |
| `gdf_path` | string | Source GDF path |

### §5a. Validation script: `create_slda/validate_features.m`

Stage-by-stage comparison of Python (notebook) vs MATLAB (`apply_processing` from `analysis_bci/matlab_simulation`) on the same calibration GDF. Produces three comparison figures — each row shows Python vs MATLAB overlaid (left) and `|diff|` with `max` annotated (right) — for a single representative trial:

| Figure | Content | Expected MAE |
|--------|---------|-------------|
| **Fig 1** | Pre-CSP ring-buffer power per band, one channel, one trial | ~5×10⁻² |
| **Fig 2** | Post-CSP linear power (sLDA-selected features, ≤8) | ~7.5×10⁻³ |
| **Fig 3** | log(post-CSP power) = sLDA input (same features) | ~1.4×10⁻² |

The residual MAE is an inherent consequence of the **chunk-alignment offset**: Python extracts windows aligned to the CF onset sample, while MATLAB's ring buffer is aligned to the nearest chunk boundary. For `chunk_size = 25`, the offset ranges from 0 to 12 samples. Both sides use ba-form `butter + lfilter / filter` with zero ICs, so the filter form is no longer a source of error.

**CAR fallback**: `validate_features.m` loads the companion YAML (same directory as the GDF for evaluation recordings; `../parameters/` sibling folder for calibration recordings). If the YAML has no `CarCfg`, the script falls back to `py.exclude_channels` from the `.mat`. Always re-run Cell 10 of the notebook before running `validate_features` on a new GDF to ensure the `.mat` carries the latest `exclude_channels`.

If Stage 1 MAE >> 0.1 → CAR mismatch (regenerate `.mat` with Cell 10). If Stage 1 MAE is small but Stage 3 MAE is large → CSP matrices differ.

Run:
```matlab
cd /home/paolo/bci_vr_ws/src/slda_bci/create_slda
validate_features   % GUI file picker → _training_features.mat
```

### Key parameters (top of notebook)

| Variable | Default (MI) | Default (CVSA) | Description |
|----------|-------------|----------------|-------------|
| `FREQ` | 500 | 500 | Sample rate (Hz) |
| `CHUNK_SIZE` | 25 | 25 | Frame size (samples) |
| `BANDS` | `[[8,12], [12,16], [16,20], [20,26]]` | `[[8,12], [10,14], [12,16]]` | Frequency bands (MI: 4 bands, CVSA: 3 bands) |
| `N_CSP_COMPONENTS` | 4 | 4 | CSP components per band |
| `EXCLUDE_CHANNELS` | `['Fp1','Fp2']` | same | Excluded from CAR mean |
| `SELECTED_CHANNELS` | 11 motor cortex channels | 8 occipito-parietal channels | CSP channel subset |
| `CVSA_INFLUENCE` | `None` | `2.5` | Seconds of CF to use per trial (`None` = all) |
| `K_FOLDS` | 6 | 6 | CV folds (5 trials/fold with 30 trials/class) |
| `FS_METHOD` | `'mrmr'` | `'mrmr'` | Feature selection: `'none'`/`'fisher'`/`'mibif'`/`'mrmr'` |
| `FS_K_TOP_MAX` | 10 | 6 | Hard cap on number of features kept |
| `FS_SCORE_FRAC` | 0.2 | 0.2 | Adaptive cap threshold (drop below `score_frac × top`) |

---

## 6. Testing

Test data and output files are all stored under `test_node_data/` in the workspace root. The logger creates `test_node_data/slda_bci/` automatically on first run.

```
test_node_data/
├── prova32ch.gdf                          ← GDF test input
├── processing_bci/
│   └── fbcsp_processing.csv              ← CSV test input (output of processing_bci CSV test)
└── slda_bci/                             ← created automatically by the logger
    ├── slda_output.csv
    ├── slda_output_first_seq.txt
    ├── slda_gdf_output.csv
    └── slda_gdf_output_first_seq.txt
```

Config files used (no duplicates — referenced from their owning package):

| File | Source |
|------|--------|
| `car.yaml` | `$(find rosneuro_filters_car)/cfg/car.yaml` |
| `csp_mi_test.yaml` | `$(find processing_bci)/cfg/csp/mi/csp_mi_test.yaml` |
| `slda_test.yaml` | `$(find slda_bci)/models/slda_test.yaml` |

### 6a. CSV test (sLDA in isolation)

Requires `test_node_data/processing_bci/fbcsp_processing.csv` — run the processing_bci CSV test first. Publishes raw mean power features directly into the sLDA node; validates only the log + sigmoid step.

```bash
roslaunch slda_bci test_node_slda.launch
# Ctrl+C → test_node_data/slda_bci/slda_output.csv + …_first_seq.txt
```

Compare with Python or MATLAB:

```python
input_mode = 'csv'
# python3 src/slda_bci/test/test_slda.py
```
```matlab
input_mode = 'csv';
test_slda   % from workspace root
```

### 6b. GDF test (full end-to-end pipeline)

Runs the complete chain: GDF replay → `rosneuro_acquisition` → `processing_fbcsp_node` → `slda.py` → logger.

```bash
roslaunch slda_bci test_node_slda_gdf.launch
# Ctrl+C → test_node_data/slda_bci/slda_gdf_output.csv + …_first_seq.txt
```

Override defaults only if needed:

```bash
roslaunch slda_bci test_node_slda_gdf.launch \
    gdf_file:=/path/to/other.gdf \
    path_slda_model:=/path/to/model.yaml
```

```
prova32ch.gdf
  → rosneuro_acquisition  (eegdev datafile plugin)
      → /neurodata  (NeuroFrame)
          → processing_fbcsp_node  (CAR + LP→HP + ring buffer + CSP)
              → /mi/eeg_fbcsp  (eeg_fbcsp)
                  → slda_node  (log + sLDA sigmoid)
                      → /mi/neuroprediction/raw  (NeuroOutput)
                          → test_logger_slda → test_node_data/slda_bci/slda_gdf_output.csv
```

Compare with Python or MATLAB:

```python
input_mode = 'gdf'
# python3 src/slda_bci/test/test_slda.py
```
```matlab
input_mode = 'gdf';
test_slda   % from workspace root
```

### Alignment and validation

Both test scripts apply the same validation strategy:

- **`first_seq`**: logger records the first received sequence number; both references skip the corresponding initial frames so comparison starts from the same state.
- **xcorr alignment**: cross-correlation on `p(class 2)` bounded at ±20 frames detects any residual startup lag and corrects it.
- **Two plots**: RAW (unaligned) and ALIGNED (lag-corrected).

### Validation results

| Test | Reference | MAE p(c₂) aligned | Notes |
|------|-----------|-------------------|-------|
| CSV | MATLAB | < 1 × 10⁻⁷ | Float32 round-trip only source of error |
| CSV | Python | < 1 × 10⁻⁷ | Identical |
| GDF | MATLAB | < 1 × 10⁻⁶ | Perfect match across full pipeline |
| GDF | Python | < 1 × 10⁻⁶ | After fixing Status channel exclusion and zero filter IC |

---

## 7. Offline simulation (`matlab_simulation`)

The `analysis_bci/matlab_simulation` package replays recorded GDF sessions through the same pipeline. Two files were updated to support the new sLDA features:

**`io/load_slda.m`** — reads `selected_feature_indices` from the YAML and converts the 0-based Python indices to 1-based MATLAB indices. Backwards-compatible: older models without the field work unchanged.

**`classifier/apply_slda.m`** — after `log(features)` and band-order reordering, applies the feature mask (`Xlog(:, selected_feature_indices)`) before multiplying by `slda.weights`. No-op when `selected_feature_indices` is empty.

No changes needed in `main_simulate.m` or `main_evaluate_metrics.m` — the feature selection is transparent to callers.

---

## 8. Dependencies

| Library | Used for |
|---------|---------|
| `processing_bci` | `eeg_fbcsp` message type, FBCSP publisher test node |
| `rosneuro_msgs` | `NeuroOutput` message |
| `scikit-learn` | `LinearDiscriminantAnalysis`, `LogisticRegression` (Platt) |
| `numpy`, `scipy` | Feature computation, sigmoid |
| `mne` | GDF loading in training notebook and Python test |
| `yaml` | Model loading |
