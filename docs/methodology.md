# Research Methodology

## Table of Contents
1. [Study Area](#study-area)
2. [Data Collection](#data-collection)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Gap-Filling Methods](#gap-filling-methods)
5. [Model Architectures](#model-architectures)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Experimental Design](#experimental-design)

---

## 1. Study Area

### Geographic Location
- **River**: Padma River, Bangladesh
- **Coordinates**: 
  - Northwest: 24.334°N, 88.757°E
  - Southeast: 23.153°N, 90.600°E
- **Area**: Approximately 5,000 km²
- **Significance**: Major distributary of the Ganges River, highly dynamic morphology

### Characteristics
- **Climate**: Tropical monsoon
- **Seasonal Variation**: 
  - Dry season (Nov-Apr): Low water levels
  - Monsoon season (May-Oct): Peak flooding
- **Morphological Features**:
  - Braided channel pattern
  - Active char-lands (mid-channel bars)
  - High erosion and accretion rates

---

## 2. Data Collection

### 2.1 Satellite Data Sources

#### Landsat Program (1988-2014)
- **Satellites**: Landsat 5 TM, Landsat 7 ETM+, Landsat 8 OLI
- **Spatial Resolution**: 30m (resampled to 60m)
- **Temporal Resolution**: 16 days
- **Spectral Bands Used**:
  - Band 2/3: Green (0.53-0.59 μm)
  - Band 5/6: SWIR1 (1.55-1.75 μm)
- **Cloud Masking**: QA_PIXEL band

#### Sentinel-2 (2015-2025)
- **Platform**: Sentinel-2A/2B
- **Spatial Resolution**: 10m-20m (harmonized to 60m)
- **Temporal Resolution**: 5 days (combined constellation)
- **Spectral Bands Used**:
  - B3: Green (560 nm, 10m)
  - B11: SWIR (1610 nm, 20m)
- **Cloud Masking**: QA60 band (cirrus and opaque clouds)
- **Data Source**: COPERNICUS/S2_SR_HARMONIZED

#### Sentinel-1 (Fallback)
- **Platform**: Sentinel-1A/1B
- **Type**: C-band SAR
- **Spatial Resolution**: 10m (resampled to 60m)
- **Polarization**: VV and VH
- **Advantages**: Cloud-independent, all-weather imaging

### 2.2 Temporal Scales

#### Yearly Composites (1988-2025)
- **Total**: 38 years
- **Purpose**: Long-term trend analysis
- **Composite Method**: Median composite of all available images per year

#### Quarterly Composites (2015-2025)
- **Total**: 44 quarters (11 years × 4)
- **Quarters**:
  - Q1: January-March (Dry season)
  - Q2: April-June (Pre-monsoon)
  - Q3: July-September (Monsoon peak)
  - Q4: October-December (Post-monsoon)
- **Purpose**: Seasonal pattern analysis

#### Bi-monthly Composites (2015-2025)
- **Total**: 66 bi-months (11 years × 6)
- **Periods**:
  - BM1: Jan-Feb
  - BM2: Mar-Apr
  - BM3: May-Jun
  - BM4: Jul-Aug
  - BM5: Sep-Oct
  - BM6: Nov-Dec
- **Purpose**: Fine-grained temporal dynamics

### 2.3 Water Detection Methods

#### MNDWI (Modified Normalized Difference Water Index)
```
MNDWI = (Green - SWIR1) / (Green + SWIR1)
```
- **Threshold**: MNDWI > 0
- **Source**: Sentinel-2, Landsat
- **Advantages**: Superior water-land discrimination compared to NDWI

#### SAR Water Detection
```
Water Mask = VV < -16 dB
```
- **Source**: Sentinel-1
- **Advantages**: Cloud-independent
- **Use Case**: Fallback when optical data insufficient

---

## 3. Preprocessing Pipeline

### 3.1 Cloud Masking
1. **Landsat**: QA_PIXEL band analysis
   - Bit 3: Cloud shadow
   - Bit 4: Cloud
2. **Sentinel-2**: QA60 band analysis
   - Bit 10: Opaque clouds
   - Bit 11: Cirrus clouds
3. **Maximum Cloud Cover**: 50%

### 3.2 Atmospheric Correction
- **Landsat**: Surface Reflectance (SR) products
  - Scale factor: 0.0000275
  - Offset: -0.2
- **Sentinel-2**: Surface Reflectance Harmonized
  - Scale factor: 0.0001

### 3.3 Harmonization
- **Spatial Resolution**: All data resampled to 60m
- **Projection**: EPSG:4326 (WGS84)
- **Resampling Method**: Nearest neighbor (for binary masks)

### 3.4 Composite Generation
- **Method**: Median composite
- **Advantages**:
  - Reduces noise
  - Mitigates remaining cloud effects
  - Preserves typical water extent

---

## 4. Gap-Filling Methods

### 4.1 Problem Statement
- Missing data due to:
  - Cloud cover
  - Sensor failures
  - Data acquisition gaps
- **Gap Percentage**: ~20% of expected imagery

### 4.2 Implemented Methods

#### Method 1: Mean Composite
```python
filled_mask = mean(neighboring_masks)
```
- **Neighbors**: 3 images before + 3 images after
- **Threshold**: 0.5

#### Method 2: Median Composite
```python
filled_mask = median(neighboring_masks)
```
- **Advantages**: Robust to outliers

#### Method 3: Linear Interpolation
```python
filled_mask = (1-w) * mask_before + w * mask_after
w = (t_target - t_before) / (t_after - t_before)
```

#### Method 4: Cubic Spline Interpolation
```python
for each_pixel:
    spline = CubicSpline(years, pixel_values)
    filled_value = spline(target_year)
```

#### Method 5: Signed Distance Field (SDF)
```python
sdf_before = distance_transform(mask_before)
sdf_after = distance_transform(mask_after)
filled_sdf = (1-w) * sdf_before + w * sdf_after
filled_mask = (filled_sdf > 0)
```
- **Advantages**: Preserves shape morphology

#### Method 6: Weighted Temporal
```python
weights = 1 / max(|year_i - target_year|, 0.5)
weights = weights / sum(weights)
filled_mask = sum(weights_i * mask_i)
```

#### Method 7: Bidirectional ConvLSTM (Best)
```python
class BiConvLSTM(nn.Module):
    forward_lstm = ConvLSTM(in_channels=1, hidden=32)
    backward_lstm = ConvLSTM(in_channels=1, hidden=32)
    output = Conv2d(hidden*2, 1, kernel=1)
```
- **Architecture**: Encoder-decoder with skip connections
- **Training**: 30 epochs, BCE loss
- **Performance**: IoU = 0.7366, Dice = 0.8477 (best among all methods)

### 4.3 Evaluation Metrics for Gap-Filling

Twenty percent of the annual series (years 2002, 2005, 2010, 2015, 2019, 2020, 2024) was removed
and reconstructed with each method. Metrics were computed over the held-out years.

| Method | IoU ↑ | Dice ↑ | Precision ↑ | Recall ↑ |
|--------|-------|--------|-------------|----------|
| Mean Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Median Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Linear Interpolation | 0.6548 | 0.7910 | 0.8913 | 0.7193 |
| Spline Interpolation | 0.6733 | 0.8042 | 0.8082 | 0.8012 |
| SDF | 0.7290 | 0.8429 | 0.8741 | 0.8161 |
| Weighted Temporal | 0.6724 | 0.8035 | 0.8306 | 0.7790 |
| **BiConvLSTM** | **0.7366** | **0.8477** | 0.8457 | 0.8511 |

The BiConvLSTM achieved the best overall performance; among classical methods the SDF approach
performed strongest (IoU = 0.7290), closely approaching deep-learning performance while
maintaining structural consistency.

---

## 5. Model Architectures

### 5.1 ConvLSTM Model

#### Architecture
```
Input: Sequence of water masks (T, H, W)
├── ConvLSTMCell 1: (1, 64, 3×3)
├── ConvLSTMCell 2: (64, 128, 3×3)
├── ConvLSTMCell 3: (128, 64, 3×3)
└── Output Conv: (64, 1, 1×1) + Sigmoid
```

#### Key Features
- **Bidirectional Processing**: Forward + Backward LSTM
- **Spatial Context**: Convolutional operations within LSTM
- **Temporal Memory**: Hidden states capture sequence patterns

#### Hyperparameters
- **Hidden Channels**: 64
- **Kernel Size**: 3×3
- **Batch Size**: 4
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Epochs**: 100

### 5.2 Swin Transformer Model

#### Architecture
```
Input: Sequence of water masks (T, H, W)
├── Patch Embedding: (4×4 patches)
├── Swin Transformer Block 1: (embed_dim=96, heads=3)
├── Patch Merging + Swin Block 2: (embed_dim=192, heads=6)
├── Patch Merging + Swin Block 3: (embed_dim=384, heads=12)
├── Decoder: (Transposed Conv + Skip Connections)
└── Output: (1, H, W) + Sigmoid
```

#### Key Features
- **Shifted Window Attention**: Efficient self-attention
- **Hierarchical Features**: Multi-scale representations
- **Position Encoding**: Relative position bias

#### Hyperparameters
- **Patch Size**: 4×4
- **Window Size**: 7×7
- **Embedding Dimension**: 96
- **Depths**: [2, 2, 6, 2]
- **Attention Heads**: [3, 6, 12, 24]
- **Learning Rate**: 1e-4
- **Batch Size**: 2 (due to memory constraints)

### 5.3 Training Strategy

#### Data Splitting
- **Training**: 70% (1988-2012 for yearly)
- **Validation**: 15% (2013-2017)
- **Testing**: 15% (2018-2025)

#### Data Augmentation
- Random horizontal flip
- Random vertical flip
- Random rotation (90°, 180°, 270°)
- **Note**: No geometric transformations that change shapes

#### Early Stopping
- **Patience**: 10 epochs
- **Monitor**: Validation IoU
- **Mode**: Maximize

---

## 6. Evaluation Metrics

### 6.1 Pixel-Level Metrics

#### Intersection over Union (IoU)
```
IoU = TP / (TP + FP + FN)
```
- **Range**: [0, 1]
- **Interpretation**: Overlap between prediction and ground truth

#### Dice Coefficient
```
Dice = 2×TP / (2×TP + FP + FN)
```
- **Range**: [0, 1]
- **Interpretation**: F1-score for binary segmentation

#### Precision & Recall
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

### 6.2 Area-Based Metrics

#### Mean Absolute Error (MAE)
```
MAE = mean(|predicted_area - actual_area|)
```
- **Unit**: km²

#### Root Mean Squared Error (RMSE)
```
RMSE = sqrt(mean((predicted_area - actual_area)²))
```
- **Unit**: km²

#### Relative Error
```
Relative Error = |predicted_area - actual_area| / actual_area × 100%
```
- **Unit**: %

### 6.3 Morphological Metrics

#### Centerline Shift
- Distance between predicted and actual river centerlines
- **Unit**: meters

#### Width Deviation
- Difference in river width measurements
- **Unit**: meters

---

## 7. Experimental Design

### 7.1 Experiments

#### Experiment 1: Gap-Filling Comparison
- **Objective**: Select best gap-filling method
- **Methods**: 7 approaches
- **Evaluation**: IoU, Dice, Precision, Recall

#### Experiment 2: Model Comparison
- **Objective**: Compare ConvLSTM vs. Swin Transformer
- **Scales**: Yearly, Quarterly, Bi-monthly
- **Metrics**: All evaluation metrics

#### Experiment 3: Temporal Scale Analysis
- **Objective**: Optimal temporal resolution
- **Scales**: Yearly, Quarterly, Bi-monthly
- **Analysis**: Prediction accuracy vs. temporal resolution

#### Experiment 4: Long-term Prediction
- **Objective**: Multi-year ahead prediction
- **Horizon**: 1, 3, 5, 10 years
- **Analysis**: Degradation of accuracy over time

### 7.2 Ablation Studies

#### ConvLSTM Ablations
1. Number of LSTM layers (1, 2, 3)
2. Hidden dimensions (32, 64, 128)
3. Bidirectional vs. Unidirectional
4. Kernel sizes (3×3, 5×5, 7×7)

#### Swin Transformer Ablations
1. Window sizes (4, 7, 14)
2. Patch sizes (2, 4, 8)
3. Number of transformer blocks
4. Attention head configurations

### 7.3 Statistical Analysis

#### Trend Analysis
- Mann-Kendall test for monotonic trends
- Theil-Sen slope estimator

#### Seasonal Decomposition
- Additive model: Observation = Trend + Seasonal + Residual
- STL decomposition (Seasonal-Trend decomposition using Loess)

#### Correlation Analysis
- Pearson correlation: Linear relationships
- Spearman correlation: Monotonic relationships
- Cross-correlation: Temporal lags

---

## 8. Computational Resources

### Hardware
- **GPU**: NVIDIA Tesla V100 32GB (Google Colab Pro)
- **RAM**: 32 GB
- **Storage**: 100 GB (Google Drive)

### Software
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **Google Earth Engine**: Python API
- **Key Libraries**: numpy, pandas, rasterio, matplotlib, seaborn

### Processing Time
- **Data Collection**: ~6 hours (all scales)
- **Gap-Filling**: ~2 hours
- **Model Training**: 
  - ConvLSTM: ~4 hours per scale
  - Swin Transformer: ~8 hours per scale
- **Evaluation**: ~1 hour

---

## 9. Limitations and Assumptions

### Limitations
1. **Cloud Cover**: Some data loss despite filtering
2. **Sensor Differences**: Landsat-Sentinel harmonization challenges
3. **Temporal Gaps**: Landsat 7 SLC-off issue (2003-present)
4. **Computational**: Memory constraints limit batch size

### Assumptions
1. **Water Detection**: MNDWI threshold of 0 is appropriate
2. **Temporal Continuity**: Gradual morphological changes
3. **Independence**: Each time step prediction independent
4. **Stationarity**: Statistical properties remain consistent

---

## 10. Validation Strategy

### Ground Truth
- High-resolution imagery (where available)
- Field survey data (limited)
- Cross-validation with alternative water detection methods

### Uncertainty Quantification
- Bootstrap confidence intervals
- Monte Carlo dropout for uncertainty estimation
- Ensemble predictions

---

**Document Version**: 1.0  
**Last Updated**: September 22, 2026  
**Authors**: Md. Kaoser Ahamed Anik, S. S. Mahmud Turza
