# employee_training_success_prediction
Machine learning project using Scikit-learn and Random Forest to predict employee training success based on experience, training hours, attendance, pre-training score, age, and department. Includes Streamlit dashboard and cohort analysis.  README.md
# Employee Training Success Prediction using Scikit-learn

A machine learning project that predicts whether an employee is likely to be **Successful** or **Needs Improvement** after completing training.

The project uses **Python, Pandas, Scikit-learn, Random Forest, Joblib, Matplotlib, and Streamlit** to build an end-to-end employee training outcome prediction system.

##  Project Overview

Employee training programs can be evaluated using factors such as attendance, training hours, previous knowledge, work experience, and department.

This project applies machine learning to these factors to predict an employee's training outcome.

The model classifies employees into two categories:

*  **Successful**
*  **Needs Improvement**

The project also includes an interactive **Streamlit web application** for individual predictions and batch/cohort analysis.

##  Dataset

The project uses a synthetic employee training dataset containing **700 employee records**.

### Features

| Feature              | Description                      |
| -------------------- | -------------------------------- |
| `age`                | Employee age                     |
| `years_experience`   | Years of professional experience |
| `training_hours`     | Total training hours completed   |
| `attendance_percent` | Training attendance percentage   |
| `pre_training_score` | Score before training            |
| `department`         | Employee department              |
| `training_result`    | Target variable                  |

### Departments

The dataset contains employees from:

* HR
* IT
* Operations
* Sales
* Finance

### Target Distribution

| Training Result   | Records |
| ----------------- | ------: |
| Successful        |     415 |
| Needs Improvement |     285 |

> **Note:** The dataset is synthetic and intended for educational and demonstration purposes.

##  Machine Learning Approach

The project uses a **Random Forest Classifier** with a Scikit-learn preprocessing pipeline.

### Preprocessing

The `department` categorical feature is transformed using:

* `OneHotEncoder`
* `handle_unknown="ignore"`

The numerical features are passed directly to the model.

### Model

A `RandomForestClassifier` is configured with:

* **200 decision trees**
* `random_state=42`
* `class_weight="balanced"`

### Train/Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

Stratified splitting is used to maintain the target-class distribution.

### Evaluation

The model is evaluated using:

* Accuracy
* Classification Report
* Confusion Matrix

A confusion matrix visualization is saved as:

```text
confusion_matrix.png
```

##  Streamlit Application

The project includes an interactive Streamlit application that provides a user-friendly interface for making predictions.

### Individual Prediction

Users can enter employee information such as:

* Age
* Years of experience
* Training hours
* Attendance percentage
* Pre-training score
* Department

The application returns:

* Predicted training result
* Model confidence
* Success likelihood
* Performance-related insights
* Learning and development recommendations

### Batch Cohort Analysis

The application can also load employee records from a CSV file and perform cohort-level analysis.

This makes the project useful for exploring training outcomes across groups of employees rather than only predicting a single employee.

##  Project Structure

```text
Employee_Training_Success_Prediction_Sklearn/
│
├── app.py
│
├── data/
│   └── employee_training_success.csv
│
├── employee_training_success_model.pkl
├── predict.py
├── train_model.py
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

##  Technologies Used

* **Python**
* **Pandas** – data loading and manipulation
* **Scikit-learn** – machine learning and preprocessing
* **Random Forest Classifier** – prediction model
* **Joblib** – model serialization
* **Matplotlib** – confusion matrix visualization
* **Streamlit** – interactive web application

##  Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Employee_Training_Success_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Train the Model

To train the Random Forest model:

```bash
python train_model.py
```

This will:

1. Load the employee training dataset.
2. Separate features and target.
3. Encode the department feature.
4. Split the data into training and testing sets.
5. Train the Random Forest classifier.
6. Evaluate the model.
7. Save the trained model.
8. Generate the confusion matrix.

The trained model is saved as:

```text
employee_training_success_model.pkl
```

##  Make a Prediction

Run:

```bash
python predict.py
```

The prediction script uses a sample employee profile and displays:

```text
Predicted Training Result: ...
Model Confidence: ...%
```

##  Run the Streamlit App

Start the interactive application with:

```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit in your browser.

##  Example Prediction Input

Example employee profile used by the prediction script:

```text
Age: 29
Years of Experience: 4
Training Hours: 45
Attendance: 92%
Pre-training Score: 72
Department: IT
```

The trained model then predicts the employee's likely training outcome and provides a confidence score.

##  Project Workflow

```text
Employee Training Dataset
          │
          ▼
    Data Preparation
          │
          ▼
   Feature Preprocessing
   ┌───────────────────┐
   │ Numeric Features  │
   │ One-Hot Encoding  │
   └───────────────────┘
          │
          ▼
 Random Forest Classifier
          │
          ▼
      Evaluation
   ┌──────┼─────────┐
   ▼      ▼         ▼
Accuracy Report  Confusion
                 Matrix
          │
          ▼
   Trained Model (.pkl)
          │
          ▼
 Individual / Cohort
      Prediction
          │
          ▼
     Streamlit App
```

##  Potential Applications

This project demonstrates how machine learning can support:

* Employee training evaluation
* Learning & Development analytics
* Training program monitoring
* Identification of employees who may need additional support
* Cohort-level training analysis
* Data-driven training recommendations

##  Disclaimer

This project uses a **synthetic dataset** and should not be used to make real-world employment, promotion, hiring, or performance decisions.

The predictions are intended for **educational and demonstration purposes only**.


If you found this project useful, feel free to ⭐ the repository.
