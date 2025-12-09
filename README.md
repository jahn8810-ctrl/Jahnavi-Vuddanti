❤️ Heart Disease Prediction using XGBoost
![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Heart%20Disease-red)
![Task](https://img.shields.io/badge/Task-Classification-yellow)

![Repo Size](https://img.shields.io/github/repo-size/jahn8810-ctrl/Jahnavi-Vuddanti)
![Last Commit](https://img.shields.io/github/last-commit/jahn8810-ctrl/Jahnavi-Vuddanti)

This project predicts the likelihood of heart disease using patient medical data.
It includes data preprocessing, model training, evaluation, and visualizations using the powerful XGBoost algorithm.

📂 Dataset

The dataset used is heart.csv (inside archive.zip).

Contains patient medical information such as:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Maximum heart rate

Exercise-induced angina

And more…

⚙️ Features & Preprocessing
🔹 Categorical Features

One-hot encoded

Handles chest pain type, fasting blood sugar, rest ECG, etc.

🔹 Numerical Features

Scaled using StandardScaler

Improves model learning and stability

🔹 Train-Test Split

80% training

20% testing

🧠 Model: XGBoost Classifier

XGBoost is chosen because it offers:

Excellent accuracy

Handles mixed numerical/categorical data

Built-in regularization

Fast and scalable performance

📊 Evaluation Metrics

The model performance is evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Feature Importance Plot

These metrics give a full understanding of how the model performs on heart disease prediction.

🧪 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Install the dependencies
pip install -r requirements.txt

3️⃣ Run the script
python heart_disease_prediction.py


Or open the Jupyter notebook:

heart_disease_xgboost.ipynb

📈 Visualizations Included

Confusion Matrix

ROC Curve

Feature Importance Chart

These help visualize model performance and interpretability.

🚀 Future Improvements (Optional but adds more value)

You can further improve the project by adding:

✔️ A Streamlit Web App

Let users input patient data and get predictions.

✔️ Model Deployment

Deploy on:

Streamlit Cloud

HuggingFace Spaces

Render

✔️ Hyperparameter Tuning

Using GridSearchCV / RandomizedSearchCV / Optuna.

✔️ SHAP Explainability

Explain why the model made each prediction.

🏁 Conclusion

This project demonstrates:

End-to-end Machine Learning workflow

Strong preprocessing

Use of XGBoost

Clean evaluation and visualizations

