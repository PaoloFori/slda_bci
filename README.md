# slda_bci

ROS node implementing a **Shrinkage Linear Discriminant Analysis (sLDA)** classifier for online BCI decoding. It subscribes to FBCSP features published by `processing_bci`, classifies each frame, and publishes class probabilities as a `NeuroOutput` message.

---

## 1. Algorithm

For each incoming `eeg_fbcsp` frame:

1. **Feature extraction**: read `data[]` from the message (raw mean power `mean(x²)` per CSP component per band), reorder to match the model's expected band order.
2. **Log transform**: `log(mean(x²))` — converts power to log-power, matching the training convention.
3. **Linear discriminant**: `score = w · x + b` where `w` is the LDA weight vector and `b` is the intercept, both loaded from the YAML model.
4. **Platt Calibration**: Apply unbiased Platt scaling to the raw decision score using parameters `platt_a` and `platt_b` loaded from the model:
   $$P(c_2) = \frac{1}{1 + e^{-(\text{platt\_a} \cdot \text{score} + \text{platt\_b})}}$$
   $$P(c_1) = 1 - P(c_2)$$
   If Platt parameters are not present in the YAML, they default to standard sigmoid scaling (`platt_a = 1.0`, `platt_b = 0.0`).
5. **Publish** `/{paradigm}/neuroprediction/raw` (`rosneuro_msgs/NeuroOutput`) with `softpredict` = `[p(c₁), p(c₂)]`.

The classifier is reconstructed at startup from `coef_` and `intercept_` stored in the YAML, using `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr')`.

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
    cv_mean_acc: 0.93
    cv_std_acc:  0.02
    slda_weights:   [[w0, w1, ..., w15]]               # [1 × n_features]
    slda_intercept: [b]                                # scalar
    slda_calibrated_weights: [platt_a]                 # Platt scaling weight coefficient
    slda_calibrated_intercept: [platt_b]               # Platt scaling bias intercept
    csp: /path/to/csp_model.yaml
```

The node reads only `classes`, `bands`, `selected_components_indices`, `slda_weights`, `slda_intercept`, `slda_calibrated_weights`, and `slda_calibrated_intercept`. The other fields are metadata saved for traceability.

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
3. **Bandpass** per band: causal Butterworth order 4, applied as **LP then HP** sequentially — matching `Fbcsp.cpp` `filters_low_[i] → filters_high_[i]`
4. Extract 1-second sliding windows (step = `CHUNK_SIZE`) during continuous feedback (event 781)
   - **Trim start alignment**: the first training window starts at `trim_start_s = -0.95 s` relative to event 781 (CF onset), containing exactly 475 samples of cue and 25 samples of feedback. This replicates the exact initial ring buffer transient that the online classifier sees at runtime.
5. **CSP** (`n_components=4`, `reg='ledoit_wolf'`, `log=True`) on `SELECTED_CHANNELS` subset per band
6. **Cross-validate** `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])` with `StratifiedGroupKFold(n_splits=4, groups=trials)` — prevents leakage from overlapping windows.
7. **Platt Calibration Fitting**: Fit an unbiased 1D Logistic Regression on the out-of-fold CV test scores (`all_scores`) to calculate unbiased calibration coefficients (`platt_a` and `platt_b`) and avoid overfitting.
8. **Fold Sigma Monitoring**: A critical check is run on the cross-validation fold variance. If the fold standard deviation exceeds 8% (`std_acc > 0.08`), the notebook prints a warning recommending to reduce `N_CSP_COMPONENTS` to `2` to stabilize feature learning and halve feature dimensionality.
9. Train final model on all data, save CSP yaml + sLDA yaml.

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

## 7. Dependencies

| Library | Used for |
|---------|---------|
| `processing_bci` | `eeg_fbcsp` message type, FBCSP publisher test node |
| `rosneuro_msgs` | `NeuroOutput` message |
| `scikit-learn` | `LinearDiscriminantAnalysis` |
| `numpy`, `scipy` | Feature computation, sigmoid |
| `mne` | GDF loading in training notebook and Python test |
| `yaml` | Model loading |
