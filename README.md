# Iris-Flower-Classification
A machine learning model for classifying Iris flowers into three species based on sepal and petal measurements.

## Project Structure
```
Iris-Classification/
│-- data/                  # Dataset files
│-- notebooks/             # Jupyter Notebooks for EDA and model training
│-- src/                   # Source code for data preprocessing, training, and evaluation
│   │-- preprocess.py      # Handles missing values, encoding, and feature scaling
│   │-- train.py           # Model training script
│   │-- evaluate.py        # Model evaluation metrics
│-- models/                # Saved trained models
│-- requirements.txt       # List of dependencies
│-- README.md              # Project documentation
```

## Dataset
Dataset: [Iris Flower Dataset](https://www.kaggle.com/arshid/iris-flower-dataset)
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Target**: Species (Setosa, Versicolor, Virginica)

## Model Selection
Random Forest Classifier is used because:
- Handles non-linearity and feature interactions well
- Provides feature importance scores
- Robust to missing values and outliers

## Evaluation Metrics
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**

## Visualizations
- Pairplot of feature distribution
- Feature importance bar chart
- Confusion matrix heatmap




