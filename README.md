# ❤️ Heart Disease Prediction using XGBoost

![Repo Size](https://img.shields.io/github/repo-size/jahn8810-ctrl/Jahnavi-Vuddanti)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-heart.csv-red)
![Task](https://img.shields.io/badge/Task-Classification-yellow)

This project predicts the likelihood of **heart disease** using patient medical data.  
It includes data preprocessing, model training, evaluation, and visualizations using the **XGBoost** algorithm.



 📂 Dataset

**File:** `heart.csv`

The dataset contains these important medical features:

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Rest ECG  
- Max Heart Rate  
- Exercise-Induced Angina  
- ST Depression  
- Slope  
- Number of Major Vessels  
- Thal  

## ⚙️ Features & Preprocessing

 🔹 Categorical Features  
- One-hot encoding  
- Applies to CP, FBS, RestECG, Slope, Thal, etc.

🔹 Numerical Features  
- Scaled using **StandardScaler**

🔹 Train-Test Split  
- 80% Training  
- 20% Testing  



🧠 Model Used — XGBoost Classifier

Reasons for choosing XGBoost:

- High accuracy  
- Handles mixed data types  
- Regularization reduces overfitting  
- Very fast training  

---

 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  
- Feature Importance Plot  

---

🧪 How to Run the Project

 1️⃣ Clone the repository
```bash
git clone https://github.com/jahn8810-ctrl/Jahnavi-Vuddanti.git
cd Jahnavi-Vuddanti
