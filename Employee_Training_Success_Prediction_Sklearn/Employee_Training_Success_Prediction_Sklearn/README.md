# Employee Training Success Prediction using Python and Scikit-learn

## Project Overview
This machine learning project predicts whether an employee will be Successful or Need Improvement after training using age, years of experience, training hours, attendance percentage, pre-training score, and department.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- OneHotEncoder
- Joblib
- Matplotlib

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts the training result for a sample employee
- `employee_training_success_model.pkl` - trained model
- `data/employee_training_success.csv` - synthetic dataset
- `confusion_matrix.png` - evaluation chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

## Output
The model predicts whether an employee is Successful or Needs Improvement.

Note: The dataset is synthetic and intended for educational purposes only.
