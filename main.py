import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Step 1: Load the dataset with error handling
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print("Dataset Loaded Successfully")
        print("Data Head:\n", data.head())
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 2: Exploratory Data Analysis (EDA)
def eda(data):
    print("\nDataset Info:")
    print(data.info())
    
    print("\nSummary Statistics:")
    print(data.describe(include='all'))  # Include all columns for descriptive stats
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Visualize correlations
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] > 1:  # Ensure at least two numeric columns
        print("\nCorrelation Matrix:")
        print(numeric_data.corr())
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()
    else:
        print("\nNot enough numeric columns for a correlation heatmap.")
    
    # Distribution of target variable
    if 'audience_rating' in data.columns:
        print("\nDistribution of Target Variable (audience_rating):")
        sns.histplot(data['audience_rating'], kde=True)
        plt.title("Distribution of Audience Rating")
        plt.show()
    else:
        print("\nTarget variable 'audience_rating' not found in the dataset.")

# Step 3: Build the model pipeline with preprocessing, hyperparameter tuning, and cross-validation
def build_pipeline(data):
    # Separate features and target
    if 'audience_rating' not in data.columns:
        raise ValueError("Target variable 'audience_rating' not found in the dataset.")
    
    X = data.drop(columns=['audience_rating'])
    y = data['audience_rating']
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()
    datetime_features = X.select_dtypes(include=[np.datetime64]).columns.tolist()
    
    # Process datetime features by extracting year, month, day, etc.
    def process_datetime(df, datetime_columns):
        for column in datetime_columns:
            df[column + '_year'] = df[column].dt.year
            df[column + '_month'] = df[column].dt.month
            df[column + '_day'] = df[column].dt.day
            df[column + '_hour'] = df[column].dt.hour
            df[column + '_minute'] = df[column].dt.minute
        return df.drop(columns=datetime_columns)
    
    # Apply datetime processing
    X = process_datetime(X, datetime_features)

    # Re-identify feature types after processing datetime columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Use ColumnTransformer to apply different pipelines to different data types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    
    # Define model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    return X, y, pipeline

# Step 4: Save the trained model pipeline
def save_pipeline(pipeline, file_name):
    if os.path.exists(file_name):
        print(f"{file_name} already exists. Skipping save.")
    else:
        joblib.dump(pipeline, file_name)
        print(f"Pipeline saved as {file_name}")

# Step 5: Run the pipeline
if __name__ == "__main__":
    # Load dataset
    data = load_data("C:/project/data/dataset.xls")
    
    if data is not None:
        # Perform EDA
        eda(data)
        
        # Build pipeline
        x, y, pipeline = build_pipeline(data)
        