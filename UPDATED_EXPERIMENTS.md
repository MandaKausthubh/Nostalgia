# Experiments.md

## ⚙️ Purpose
This document provides **explicit step-by-step instructions** for the coding agent to:
1. Load datasets sequentially.
2. Apply **nostalgia-based training** correctly.
3. Replace model classification heads where necessary.
4. Train and evaluate models under both **ImageNet-style** and **Cross-Domain** setups.
5. Compute and log all required metrics.
6. Save all results and visualizations in structured directories.

All tasks must be executed **in the given order**, and **nostalgia-based training must only be applied** where instructed.

---

## 🧩 1. DATASET SETUP (Sequential Loading Order)

### Step 1 — ImageNet-style block
1. Load **TinyImageNet** (train + val).  
   - Use this as the *first training dataset*.
   - **Do NOT** apply nostalgia-based loss here (no Hessian available for pretraining data).
   - Input resolution: `64×64` → resize to `224×224`.
   - Classes: 200.

2. Load **ImageNet-V2** and **ImageNet-ODD** for evaluation only.  
   - Both must use `224×224` images.
   - Classes: 1000.
   - These datasets serve to test *generalization* and *OOD robustness* after fine-tuning on TinyImageNet.

**Normalization:** Use ImageNet statistics:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
````

---

### Step 2 — Cross-Domain block

After completing all ImageNet-family experiments, sequentially load:

1. **CIFAR-10** (train/test) → 32×32, 10 classes.
2. **CIFAR-100** (train/test) → 32×32, 100 classes.
3. **Caltech-101** (train/test) → resize to 224×224, 101 classes.
4. **Caltech-256** (train/test) → resize to 224×224, 256 classes.

**Normalization:**

* CIFAR datasets use CIFAR statistics:
  `mean = [0.5071, 0.4867, 0.4408]`
  `std  = [0.2675, 0.2565, 0.2761]`
* Caltech datasets use ImageNet statistics.

---

## 🧠 2. MODEL INITIALIZATION AND HEAD REPLACEMENT

The coding agent must dynamically replace the classification head each time the dataset changes:

```python
def replace_classification_head(model, num_classes):
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model
```

This ensures the output layer dimensionality matches the dataset’s class count.

---

## 🧩 3. EXPERIMENTAL PIPELINE (Sequential Execution Plan)

The agent must execute the following blocks **in order**, logging results for each subtask.

---

### **BLOCK A — ImageNet-family Experiments**

**Objective:** Evaluate in-domain robustness and generalization (no nostalgia applied here).

#### A1. Fine-tune pretrained models on TinyImageNet

* Models:

  * `ResNet-50` (ImageNet-pretrained)
  * `ViT-B/16` (ImageNet-pretrained)
* Replace classifier head with 200-class head.
* Train on TinyImageNet for **50 epochs**.
* Optimizers:

  * ResNet → SGD (lr=0.01, momentum=0.9)
  * ViT → AdamW (lr=1e-4)
* Scheduler: cosine decay + 5% warmup
* Batch size: 128
* Weight decay: 0.01
* Label smoothing: 0.1
* Augmentations: RandomResizedCrop(224), RandomHorizontalFlip, RandAugment

#### A2. Evaluate on ImageNet-V2 and ImageNet-ODD

* Load trained checkpoint from A1.
* Evaluate accuracy and OOD robustness metrics on:

  * **ImageNet-V2** (same domain)
  * **ImageNet-ODD** (out-of-domain)
* Save all evaluation results.

⚠️ **Nostalgia training is skipped for A1 and A2**, as no pretraining Hessian exists for the ImageNet-pretrained models.

---

### **BLOCK B — Cross-Domain Experiments**

**Objective:** Analyze domain-shift generalization and nostalgia-based memory preservation.

#### B1. Train baseline models (no nostalgia)

* Models:

  * `ResNet-18` (scratch) for CIFAR-100 (32×32)
  * `ResNet-50` (ImageNet-pretrained) for Caltech-256 (224×224)
* Replace classification head with appropriate class count.
* Optimizer: SGD (lr=0.1 → cosine decay, momentum=0.9)
* Epochs: 100 (scratch) / 50 (finetune)
* Weight decay: 5e-4
* Batch size: 128
* Augmentations: RandomCrop, HorizontalFlip, Cutout

#### B2. Apply nostalgia-based training

After completing baseline training, run nostalgia-based fine-tuning for **Cross-Domain tasks only** (B1).

* Use nostalgia loss that incorporates the Hessian computed from **the immediately preceding task**.
* Update model weights using:

  ```
  L_total = L_ce + λ * L_nostalgia
  ```

  where:

  * `L_ce` = standard cross-entropy loss
  * `L_nostalgia` = quadratic penalty using saved Hessian approximation
  * `λ` = nostalgia weight hyperparameter (suggested range 0.1–1.0)
* Ensure the Hessian is computed after the first baseline (B1) task.

Each time the dataset switches (CIFAR → Caltech), the model must:

1. Load the previously trained checkpoint.
2. Apply nostalgia-based fine-tuning using saved Hessian from prior task.

---

## 📊 4. METRICS AND LOGGING REQUIREMENTS

The agent must compute **all** the following metrics for each dataset and store them in `.json` files.

| Metric                      | Description                                | Applies to              | Notes                            |
| --------------------------- | ------------------------------------------ | ----------------------- | -------------------------------- |
| **Top-1 Accuracy**          | Primary classification accuracy            | All datasets            | Report mean ± std (3 seeds)      |
| **Top-5 Accuracy**          | Secondary accuracy metric                  | ImageNet-style datasets |                                  |
| **AUROC**                   | OOD detection via softmax confidence       | ImageNet-ODD            |                                  |
| **ECE**                     | Expected Calibration Error                 | All                     | Evaluate calibration reliability |
| **Precision / Recall / F1** | Class balance metrics                      | CIFAR / Caltech         | macro-averaged                   |
| **Per-class Accuracy**      | Class-level insight                        | All                     | visualize top-10 hardest classes |
| **t-SNE / UMAP embeddings** | Feature space visualization                | All                     | last hidden layer                |
| **Loss Curves**             | Training vs validation loss                | All                     | plotted per-epoch                |
| **Nostalgia Penalty Curve** | Magnitude of nostalgia regularization term | Nostalgia runs only     | monitor stability                |

All metrics must be logged to:

* TensorBoard and/or wandb under experiment group names:

  * `"imagenet_family/*"`
  * `"cross_domain/*"`
  * `"cross_domain_nostalgia/*"`

And saved under:

```
results/
 ├── imagenet_family/
 │    ├── tinyimagenet_resnet50.json
 │    ├── imagenetv2_eval.json
 │    └── imagenetodd_eval.json
 ├── cross_domain/
 │    ├── cifar100_baseline.json
 │    ├── caltech256_baseline.json
 │    └── nostalgia/
 │         ├── cifar100_nostalgia.json
 │         └── caltech256_nostalgia.json
 └── visualizations/
      ├── accuracy_curves.png
      ├── ood_roc.png
      ├── calibration_hist.png
      ├── tsne_embeddings.png
      ├── nostalgia_loss_curve.png
```

---

## 🧮 5. ABLATION STUDIES

Each ablation should vary **exactly one factor** at a time.

| Ablation             | Variable                         | Description                             |
| -------------------- | -------------------------------- | --------------------------------------- |
| **Pretraining**      | Pretrained vs Scratch            | Effect of ImageNet initialization       |
| **Model Capacity**   | ResNet-18 / ResNet-50 / ViT-B/16 | Architecture comparison                 |
| **Augmentation**     | Standard vs Strong               | Impact of data augmentation             |
| **Nostalgia Weight** | λ ∈ {0.0, 0.1, 0.5, 1.0}         | Trade-off between memory and plasticity |
| **Resolution**       | 32×32 vs 224×224                 | Effect of input size scaling            |

Each ablation must produce separate `.json` results under:

```bash
results/ablations/<ablation_name>/<dataset>/<model>.json
```

---

## ✅ 6. EXECUTION SUMMARY

**The agent must:**

1. Execute tasks strictly in the order listed above.
2. Skip nostalgia for the first ImageNet-based training task (TinyImageNet fine-tune).
3. Save model checkpoints and Hessians after each completed task.
4. Apply nostalgia-based fine-tuning only during Cross-Domain experiments.
5. Log all metrics and produce visualizations.
6. Save all metrics as `.json` for later aggregation.

---

## 🧾 Final Notes

* All datasets and models must use reproducible random seeds.
* Each training run should be repeated **three times** to compute mean ± std.
* Visualizations must include:

  * Accuracy curves
  * OOD ROC
  * Calibration histograms
  * t-SNE feature maps
  * Nostalgia loss evolution plots

This sequential structure ensures consistent, reproducible, and interpretable results across both experiment blocks.
