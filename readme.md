# KD-TL-Seismic-Response-Prediction

### Knowledge Distillation–Based Transfer Learning Framework for Seismic Response Prediction of Urban Building Clusters

This repository provides the open-source trained models, code, and data used in our paper. It includes an example demonstration showing how to run the provided scripts to reproduce Figure 9, which compares the predictive performance of four seismic response models in terms of their correlation coefficient (r), coefficient of determination (R²), and mean squared error (MSE). The overall methodological framework is illustrated in Figure 1, and the full project—including model documentation and extended tools—will be released upon paper acceptance.

Acknowledgment: The measured building-response data used in this study were obtained from the Center for Engineering Strong Motion Data (CESMD) database.

---

## 📊 Included Models

1. **Pretrained model**  
   Trained on simulation-based source-domain data to learn general structural dynamic characteristics.

2. **Directly transferred pretrained model**  
   Fine-tuned on limited target-domain data without knowledge distillation.

3. **KD-based TL model**  
   Knowledge distillation–enhanced transfer-learning model achieving improved generalization and robustness.

4. **Direct-training baseline model**  
   Trained from scratch on target-domain data only, serving as a performance baseline.

---

## 📁 Repository Structure

```
KD-TL-Seismic-Response-Comparison
│
├── data/
│ ├── building_response_testset.csv # Test set (input features and ground-truth responses)
│ ├── pretrained_model_results.csv # Metrics: r, R², MSE for the pretrained model
│ ├── direct_transferred_model_results.csv # Metrics for directly transferred pretrained model
│ ├── kd_based_tl_model_results.csv # Metrics for KD-based TL model
│ ├── direct_training_baseline_results.csv # Metrics for direct-training baseline model
│ └── metrics_summary.csv # Combined summary file for plotting
│
├── models/
│ ├── pretrained_model.pth # Pretrained model weights
│ ├── direct_transferred_model.pth # Directly transferred model weights
│ ├── kd_based_tl_model.pth # KD-based TL model weights
│ └── direct_training_baseline.pth # Direct-training baseline weights
│
├── scripts/
│ ├── evaluate_models.py # Compute r, R², and MSE metrics from test data
│ ├── plot_fig9.py # Generate Figure 9 comparison plot
│ └── utils.py # Data-loading and helper functions
│
├── figures/
│ └── Fig9_model_comparison.png # Output figure (bar chart comparison)
│
├── LICENSE
└── README.md
```
---

## ⚙️ Installation and Reproduction Steps

### 1. Install Dependencies
This repository requires Python ≥ 3.8 and the following packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
### 2. Evaluate or Load Results
To recompute the model performance metrics (r, R², MSE):

```bash
python scripts/evaluate_models.py
```
This script uses data/building_response_testset.csv for evaluation and saves results to the corresponding model result files in data/.

### 3. Generate Figure 9
```bash
python scripts/plot_fig9.py
```
The resulting plot will be saved as: figures/Fig9_model_comparison.png

### 📈 Expected Output
Figure 9 illustrates the comparative performance of four models using color-coded bars:

|Color|	Model	|Description|
|--- |--- |--- |
|🟦 Blue|	Pretrained model|	Source-domain model trained on simulated data|
|🟩 Green|	Directly transferred pretrained model|	Source model fine-tuned on target data|
🟥 Brown|	KD-based TL model|	Knowledge-distillation-enhanced transfer model|
|🟧 Orange|	Direct-training baseline model|	Model trained from scratch on target data|

Metric bars correspond to:r — Pearson correlation coefficient
R² — Coefficient of determination
MSE — Mean squared error

---

## 📘 Data Availability
All preprocessed test data, model weights, and result files needed to reproduce Figure 9 are provided in this repository.
For transparency and reproducibility, each model’s evaluation outputs are saved as CSV files in the /data folder.

## 🧩 License
This project is released under the MIT License.
Users are encouraged to reuse and extend this code for research and educational purposes with proper citation of the associated publication.

## 🧠 Citation
If you use this repository, please cite our paper:

A knowledge distillation-based transfer learning framework for peak seismic response prediction of urban building clusters

Submitted to Engineering Structures, 2026.

## 📞 Contact
For questions, collaborations, or bug reports, please contact:
Qingle Cheng — Beijing University of Civil Engineering and Architecture
✉️ Email: chengqingle@bucea.edu.cn

© 2026 — KD‑TL‑Seismic‑Response Project