```markdown
# FP-GAN: Fixed Parity GAN via Parity-Guided Sliding Window KDE

Official PyTorch implementation of **FP-GAN (Fixed Parity Generative Adversarial Network)**, utilizing an efficient **Sliding Window Kernel Density Estimation (SW-KDE)** loss function for localized, high-precision distribution matching.

This framework is designed for high-fidelity image synthesis under heavily resource-constrained settings, reducing algorithmic GPU compute overhead significantly compared to standard regularized architectures.

---

## 🚀 Key Framework Features

* **ParityNet Bounding:** Utilizes a lightweight auxiliary Multi-Layer Perceptron (MLP) mapping unbounded discriminator scores to a bounded probability space $[0, 1]$, stabilizing structural training gradients without demanding second-order derivatives or expensive gradient penalties ($WGAN-GP$).
* **Sliding Window KDE Loss:** Replaces standard global averaging ($MSE$) with a localized density alignment matching system computed in log-linear time ($\mathcal{O}(N \log N)$), effectively serving as a structural regularizer against distant feature space outliers.
* **Green AI Efficiency:** Tailored to achieve competitive convergence profiles on consumer-grade hardware and standard cloud execution configurations in under one hour of absolute training duration.

---

## ⚙️ Core Prerequisites & Environment Setup

Ensure your local or remote workspace is configured with Python 3.8+ and an active CUDA acceleration environment. 

### Dependencies Installation
Install the necessary processing modules via `pip`:

```bash
pip install torch torchvision numpy tqdm lpips torch-fidelity kagglehub matplotlib pillow

```

*Note: The script features dynamic environment checking for `kagglehub` to handle automatic runtime asset acquisition, along with fallback systems if running inside structured Jupyter notebook contexts.*

---

## 📂 Dataset Architecture

The framework handles image ingestion using a high-throughput, un-nested `FlatImageFolder` mapping architecture. By default, it targets the **Animals-256x256** dataset environment (`orionrid/animals-256x256`).

* **Automated Execution:** If run on Kaggle, the pipeline automatically checks for pre-mounted paths at `/kaggle/input/animals-256x256`. If run locally, `kagglehub` pulls the archive dynamically to your localized cash directory.
* **Manual Mapping Configuration:** To drop in a custom structural image target, adjust the `TRAIN_DIR` string pointing directly toward your absolute disk paths.

---

## 🛠️ Hyperparameter Configurations

Fine-tune internal settings inside the root config block:

```python
IMG_SIZE = 128            # Absolute height/width dimension mapping
BATCH_SIZE = 32           # Optimization baseline
TOTAL_ITERATIONS = 25000  # Termination baseline
KDE_BANDWIDTH = 0.5       # RBF Gaussian variant width scaling value
WINDOW_RADIUS = 2         # Sparse localized neighbors matrix footprint (2 Left + 2 Right)
FREEZE_PERCEPTRON = True  # Locks target goalpost bounds for the loss landscape

```

---

## 💻 Execution Protocol

Launch the standard distribution optimization execution pipeline directly via CLI:

```bash
python train.py

```

### Production Workflows Pipeline Flow

```
[Data Input] ➔ [Lightweight DCGAN Backbone] ➔ [ParityNet Score Bounding] ➔ [Sorted Local SW-KDE Matrix Convergence Check]

```

---

## 📊 Evaluation Assets & Checkpoint Artifacts

The application generates tracking spaces during run progression to monitor data state metrics without pausing active GPU scheduling loops:

* 📂 `generated_images_fp_SlidingWindowKDE_R2_128/` — Contains grid layouts of fake distributions rendered out at predefined evaluation periods.
* 📂 `checkpoints_fp_SlidingWindowKDE_R2_128/` — Stores weights configurations. The file `best_G.pth` updates automatically whenever a lower **Fréchet Inception Distance (FID)** baseline is verified.
* 📊 **Real-time Metrics Tracking:** Evaluates generative distributions over `5000` image metrics pools checking **FID**, **KID**, **IS**, and **LPIPS Diversity Score tracking layers** via the `torch-fidelity` execution backend.

```

```
