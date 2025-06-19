
# 💡 Customer Churn Prediction – Deep Learning Project

Welcome to the **Customer Churn Prediction Deep Learning Project**. This repository hosts two interactive **Streamlit** applications powered by **deep learning models** to predict:

* The **probability of customer churn**.
* The **estimated salary** of bank customers.

Developed using **TensorFlow**, **scikit-learn**, and **Python 3.11** within an **Anaconda** environment, this project leverages the `Churn_Modelling.csv` dataset to provide data-driven insights via intuitive web interfaces.

---

## 🚀 Project Highlights

This project includes **two Streamlit applications**:

1. **Churn Prediction App** (`app.py`)
   → Predicts the likelihood of a customer leaving the bank based on their demographic and financial profile.

2. **Estimated Salary Prediction App** (`app2.py`)
   → Predicts a customer’s estimated salary using key attributes, including churn status.

Both apps utilize **pre-trained TensorFlow models** with data preprocessing handled through `LabelEncoder`, `OneHotEncoder`, and `StandardScaler` from scikit-learn. The solution is designed to be **user-friendly, modular**, and **deployment-ready**.

---

## 📊 Dataset Overview

The project is based on the **Churn\_Modelling.csv** dataset, which includes **10,000 customer records** and the following features:

| Feature           | Description                                         |
| ----------------- | --------------------------------------------------- |
| `RowNumber`       | Unique row identifier                               |
| `CustomerId`      | Unique customer ID                                  |
| `Surname`         | Customer's surname                                  |
| `CreditScore`     | Credit score (numeric)                              |
| `Geography`       | Country (France, Spain, Germany)                    |
| `Gender`          | Gender (Male, Female)                               |
| `Age`             | Age (numeric)                                       |
| `Tenure`          | Years with the bank                                 |
| `Balance`         | Account balance                                     |
| `NumOfProducts`   | Number of bank products used                        |
| `HasCrCard`       | Credit card ownership (0 = No, 1 = Yes)             |
| `IsActiveMember`  | Active membership status (0 = Inactive, 1 = Active) |
| `EstimatedSalary` | Estimated salary                                    |
| `Exited`          | Churn flag (0 = Stayed, 1 = Churned)                |

This dataset was used for training and preprocessing the models. The trained models are integrated into the Streamlit apps for real-time predictions.

---

## 📈 Model Performance

### 🔹 Churn Prediction Model (`model.h5`)

* **Training Loss:** 0.3214
* **Training Accuracy:** 86.59%
* **Test Loss:** 0.2955
* **Test Accuracy:** 87.50%
* **Validation Loss:** 0.3474
* **Validation Accuracy:** 85.90%

> 📌 *Note: While the model demonstrates strong performance, the slight difference in validation loss may suggest minor overfitting.*

### 🔹 Salary Prediction Model (`regression_model.h5`)

* A TensorFlow-based regression model.
* Performance metrics are available in training logs (not displayed here for brevity).

---

## 🛠️ Prerequisites

Ensure the following tools and files are available before running the project:

* Python 3.11
* Anaconda (recommended)
* Pre-trained models:

  * `model.h5` (churn prediction)
  * `regression_model.h5` (salary prediction)
* Preprocessing files:

  * `label_encoder_gender.pkl`
  * `onehotencoder_geography.pkl`
  * `scaler.pkl`

---

## 📦 Setup & Installation

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/HamzaImtiaz03/Customer-Churn-Prediction-DL-Project.git
cd Customer-Churn-Prediction-DL-Project
```

### 🔹 Step 2: Create and Activate Anaconda Environment

```bash
conda create -n churn_prediction python=3.11
conda activate churn_prediction
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ In case of package compatibility issues (especially TensorFlow), use:

```bash
conda install pandas numpy seaborn matplotlib scikit-learn streamlit
pip install tensorflow==2.15.0 tensorboard scikeras ipykernel
```

---

## 🚀 Launching the Applications

### 🔸 Churn Prediction App

```bash
streamlit run app.py
```

* Open in browser at: `http://localhost:8501`
* Input customer details to predict churn probability.

### 🔸 Estimated Salary Prediction App

```bash
streamlit run app2.py
```

* Enter customer features (including churn status) to estimate their salary.

---

## 🎯 Key Features

* **🧠 Deep Learning Models**: Pre-trained using TensorFlow on real-world data.
* **🖥️ Interactive UI**: Sliders and dropdowns for seamless input experience.
* **🔍 Churn Probability**: Outputs a confidence score with "likely to churn" logic.
* **💰 Salary Estimation**: Predicts numeric salary based on inputs.
* **⚙️ Robust Preprocessing**: Encoders and scalers manage both categorical and numerical data.

---

## 📋 Project Dependencies

Core dependencies include:

* `tensorflow==2.15.0`: Deep learning framework
* `scikit-learn`: Preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)
* `streamlit`: Web application interface
* `pandas`, `numpy`: Data handling
* `seaborn`, `matplotlib`: Visualization (optional)
* `tensorboard`, `scikeras`, `ipykernel`: Training and environment tools

Refer to `requirements.txt` for full list.

---

## 🔍 Additional Notes

* All model and preprocessing files must be in the **project root directory**.
* The dataset (`Churn_Modelling.csv`) is **not used directly** in the apps but was essential during training.
* Preprocessing logic can be found in both `app.py` and `app2.py` for customization.
* If TensorFlow errors arise in Python 3.11, try installing via conda or review version compatibility.

---

## 🤝 Contributing

We welcome contributions from the community!

1. **Fork** this repository
2. **Create a branch**: `git checkout -b feature-branch`
3. **Commit your changes**: `git commit -m "Add new feature"`
4. **Push to the branch**: `git push origin feature-branch`
5. **Open a Pull Request**

For bug reports or feature requests, please use **GitHub Issues**.

---

## 📬 Contact

For questions, suggestions, or support:

* Open an issue on GitHub
* Contact the maintainer: **Hamza Imtiaz**
* GitHub Profile: [HamzaImtiaz03](https://github.com/HamzaImtiaz03)

---
