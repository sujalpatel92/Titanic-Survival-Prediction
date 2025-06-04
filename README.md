# Titanic-Survival-Prediction

This repository provides a hands-on guide to the complete Machine Learning workflow using the classic Titanic dataset. It covers essential steps from data preprocessing and feature engineering to model training and evaluation, helping you build a solid understanding of how to approach a binary classification problem.

## Table of Contents

-   [Project Overview](#project-overview)
-   [ML Workflow Explained](#ml-workflow-explained)
-   [Repository Structure](#repository-structure)
-   [Setup and Installation](#setup-and-installation)
-   [How to Run](#how-to-run)
-   [Expected Outcomes and What You Will Learn](#expected-outcomes-and-what-you-will-learn)

## Project Overview

The goal of this project is to predict the survival of passengers on the Titanic based on various features such as age, gender, passenger class, and more. This is a binary classification task, where the model will predict whether a passenger survived (1) or not (0). We will walk through the entire machine learning pipeline, from understanding the raw data to evaluating the performance of different models.

## ML Workflow Explained

This project demonstrates the following key stages of a typical Machine Learning workflow:

1.  **Data Preprocessing**: Cleaning and preparing the raw data for model consumption. This includes handling missing values, transforming data types, and dealing with outliers.
2.  **Feature Engineering**: Creating new features from existing ones to improve model performance and capture more complex relationships in the data.
3.  **Data Splitting**: Dividing the dataset into training and testing sets to evaluate the model's performance on unseen data.
4.  **Model Selection & Training**: Choosing appropriate machine learning algorithms and training them on the prepared training data.
5.  **Model Evaluation**: Assessing the trained models' performance using various metrics and visualizations to understand their strengths and weaknesses.

## Repository Structure

```
Titanic-Survival-Prediction/
├── README.md
├── requirements.txt
├── data/
│   └── train.csv   (User will place this here)
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Feature_Engineering_and_Preprocessing.ipynb
│   └── 03_Model_Training_and_Evaluation.ipynb
└── src/
├── preprocess.py
├── train_model.py
└── evaluate_model.py
```

## Setup and Installation

To set up your environment and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sujalpatel92/Titanic-Survival-Prediction.git](https://github.com/sujalpatel92/Titanic-Survival-Prediction.git)
    cd Titanic-Survival-Prediction
    ```
    (Note: Replace `sujalpatel92` with your actual GitHub username or the repository's path if you've forked it.)

2.  **Download the Dataset:**
    Download the `train.csv` file from the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic/data).

3.  **Place the dataset:**
    Create a `data/` directory inside the `Titanic-Survival-Prediction/` folder and place the downloaded `train.csv` file inside it.
    ```
    Titanic-Survival-Prediction/
    └── data/
        └── train.csv
    ```

4.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

You can run this project using the Jupyter notebooks or by executing the Python scripts in the `src/` directory.

### Running with Jupyter Notebooks (Recommended for Learning)

The notebooks provide a step-by-step walkthrough of the ML workflow with explanations and visualizations.

1.  Start Jupyter Lab or Jupyter Notebook from the root of the repository:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
2.  Navigate to the `notebooks/` directory.
3.  Open and run the notebooks in the following order:
    -   `01_Exploratory_Data_Analysis.ipynb`
    -   `02_Feature_Engineering_and_Preprocessing.ipynb`
    -   `03_Model_Training_and_Evaluation.ipynb`

### Running with Python Scripts (for Modularity Demonstration)

The `src/` directory contains modular Python scripts that encapsulate the core logic. You can run these scripts to see how the notebook steps can be organized into functions.

*Note: These scripts are provided as a demonstration of modularity and are designed to be run sequentially or integrated into a larger pipeline. They do not have command-line interfaces.*

For example, to run the preprocessing:
```python
# You would typically integrate these functions into a main script or pipeline
# For demonstration purposes, you can open a Python interpreter or create a temporary script:

import pandas as pd
from src.preprocess import preprocess_titanic
from src.train_model import train_classifier
from src.evaluate_model import evaluate_classifier
from sklearn.model_selection import train_test_split

# Example usage:
df = pd.read_csv('data/train.csv')
processed_df = preprocess_titanic(df)
# You might save this processed_df to a file and then load it for training
# processed_df.to_csv('data/processed_data.csv', index=False)

X = processed_df.drop('Survived', axis=1)
y = processed_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model
lr_model = train_classifier(X_train, y_train, model_type='LogisticRegression')
print("\n--- Logistic Regression Evaluation ---")
evaluate_classifier(lr_model, X_test, y_test)

# Train a Random Forest model
rf_model = train_classifier(X_train, y_train, model_type='RandomForestClassifier')
print("\n--- Random Forest Evaluation ---")
evaluate_classifier(rf_model, X_test, y_test)
```

## Expected Outcomes and What You Will Learn
By working through this project, you will:

- Gain practical experience with data loading, cleaning, and transformation.
- Understand how to perform exploratory data analysis to uncover insights.
- Learn various feature engineering techniques to enhance model performance.
- Master the process of splitting data for training and testing.
- Be able to select, train, and evaluate different classification models.
- Interpret key evaluation metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- Develop skills in visualizing model performance, including confusion matrices and ROC curves.
- See how to structure a machine learning project for better organization and modularity.

```python
# requirements.txt
print("pandas~=2.0.0")
print("numpy~=1.24.0")
print("scikit-learn~=1.2.0")
print("matplotlib~=3.7.0")
print("seaborn~=0.12.0")
print("jupyterlab~=4.0.0")
```
