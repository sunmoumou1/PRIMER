# PRIMER

## 📜 Abstract
Precipitation remains one of the most challenging climate variables to observe and predict. Existing datasets face intricate trade-offs: gauges are relatively trustworthy but sparse, satellites provide near-global coverage with retrieval uncertainties, and numerical models offer physical consistency but are biased. Here we introduce PRIMER (Precipitation Records Infinite MERging), a framework that fuses these complementary sources. PRIMER employs a coordinate-based diffusion model that learns from arbitrary spatial locations and associated intensity values, enabling seamless integration of gridded data and irregular gauge observations. Through two-stage training—first learning large-scale patterns, then refining with gauge measurements—PRIMER captures both large-scale structure and local precision. Once trained, it can correct biases in existing datasets—yielding significant error reductions at most gauge sites—and downscale reanalysis. In addition, by combining background estimates with extra gauges, it produces analysis fields that further reduce errors. All tasks are achieved through posterior sampling utilizing the prior obtained by fusing multi-source records. Crucially, it generalizes without retraining, correcting biases in operational forecasts and downscaling future scenario precipitation fields. This shows how generative AI elevates imperfect data into strength.


---

## 🛠️ Environment Setup

To create the conda environment, run:

```bash
conda env create --name PRIMER --file requirements.yml
conda activate PRIMER
```

**CUDA information:**

We used:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 11.7, V11.7.64
with 2 × NVIDIA A100-SXM4-40GB
```

---

## 🚀 Training

### Single-source Training

To train the model using a single data source, such as only gauge observations:

```bash
python train_with_multiple_gpu.py
```

Refer to this script if you want to train PRIMER on your own single-source dataset.  
We demonstrate how PRIMER can be trained on **discrete sparse gauge observations** in this mode.

---

### Multi-source Training

To train the model using **multiple data sources** (e.g., IMERG, ERA5, and gauges):

```bash
python train_with_multiple_gpu_mutiple_sources.py
```

This script shows how to integrate various sources simultaneously.

---

## 🔍 Inference

We provide implementations of posterior sampling methods, including **SDEdit** and **Inpainting**.

You can find the inference algorithms implemented in:

```
diffusion_class/diffusion.py
```

These methods allow for flexible sampling under partial observation constraints, and can be directly applied to whatever readers reserach interests.

---

## 📦 Dataset Information

### Sources:

- **ERA5**: Downloaded from [Copernicus CDS](https://cds.climate.copernicus.eu). Fast access also available via [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2).
- **IMERG**: Downloaded from [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary?keywords=imerg).
- **Gauge Observations**: Provided by Wentao Li (China Meteorological Administration, CMA).  
  *Note: Full dataset not publicly available due to licensing. A small debug subset is provided.*

### Data Preprocessing:

Please refer to the procedures outlined in our paper for full data preprocessing pipelines.

We provide a sample set of gauge observations for debugging under the folder:

```bash
process_gauges_before/
```

---

## 📁 Repository Structure

```
.
├── code/                                    # Core training and inference code
│   ├── configs/training_config.py           # Default hyperparameter settings
│   ├── diffusion_class/                     # Diffusion model core + SDEdit/Inpainting samplers
│   ├── experiment/generate_samples.py       # Posterior sampling demo script
│   ├── models/                              # SparseConvResBlock/UNO architectures and blocks
│   ├── utils/                               # Data loading, resampling, optimizer, and training helpers
│   ├── process_gauges_before/               # Debug gauge sample data and exploration notebook
│   ├── train_with_multiple_gpu.py           # Single-source training entry
│   ├── train_with_multiple_gpu_mutiple_sources.py # Multi-source training entry
│   ├── dct_util.py                          # Discrete cosine transform utilities
│   └── remove_pycache.py                    # Clean cached Python artifacts
├── data/                                    # Example ERA5/IMERG/gauge/HRES sample arrays
├── environment/requirements.yml             # Conda environment specification
```



---


## 🙋 Contact & Contribution

For questions related to collaboration, contact the authors (ssc23@mails.tsinghua.edu.cn).

---

## 💡 Acknowledgments

We gratefully acknowledge the data sources and support from:
- Copernicus Climate Data Store (CDS)
- NASA GES DISC
- WeatherBench2 archive
- China Meteorological Administration (CMA)

---
