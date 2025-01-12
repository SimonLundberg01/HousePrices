import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# Define the feature categories
features = {
    "Categorical": [
        "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", 
        "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", 
        "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", 
        "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "BsmtQual", 
        "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", 
        "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", 
        "GarageType", "GarageFinish", "PavedDrive", "PoolQC", "Fence", 
        "MiscFeature", "SaleType", "SaleCondition", "ExterQual", "ExterCond","HeatingQC",
        "KitchenQual","GarageQual","GarageCond"
    ],
    "Continuous": [
        "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", 
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 
        "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageYrBlt", "GarageArea", 
        "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", 
        "PoolArea", "MiscVal", "SalePrice", "YrSold"
    ],
    "Ordinal": [
        "OverallQual", "OverallCond" , "Fireplaces", "GarageCars", "KitchenAbvGr",
        "BsmtHalfBath", "FullBath", "HalfBath", "BsmtFullBath", "BedroomAbvGr", "TotRmsAbvGrd",
    ]
}
conversion_list = [
    ("MasVnrType", "None"), ("BsmtQual", "NA"), ("Electrical", "SBrkr"),
    ("BsmtCond", "TA"), ("BsmtExposure", "No"), ("BsmtFinType1", "No"),
    ("BsmtFinType2", "No"), ("CentralAir", "N"), ("Condition1", "Norm"),
    ("Condition2", "Norm"), ("ExterCond", "TA"), ("ExterQual", "TA"),
    ("FireplaceQu", "NA"), ("Functional", "Typ"), ("GarageType", "No"),
    ("GarageFinish", "No"), ("GarageQual", "NA"), ("GarageCond", "NA"),
    ("HeatingQC", "TA"), ("KitchenQual", "TA"), ("MSZoning", "None"),
    ("Exterior1st", "VinylSd"), ("Exterior2nd", "VinylSd"), ("SaleType", "WD")
]

conditions = {
    'LotFrontage': 200, 'LotArea': 100000, 'MasVnrArea': 1200,
    'BsmtFinSF1': 3000, 'BsmtFinSF2': 1200, 'TotalBsmtSF': 4000,
    '1stFlrSF': 3500, 'GrLivArea': 4000, 'WoodDeckSF': 800,
    'OpenPorchSF': 450, 'EnclosedPorch': 400, 'MiscVal': 4000
}

verbose = False  # Set to False to suppress output
run_gridsearch = False 

# Function to handle missing values (NaN) based on predefined rules
def none_transform(df):
    for col, new_str in conversion_list:
        if col in df.columns:
            df[col] = df[col].fillna(new_str)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(0)
    return df
# Function to explore the data
def explore_data(df_train):
    if verbose:
        print("Basic Information about the Dataset:")
        print(df_train.info())
        print("\nSummary Statistics:")
        print(df_train.describe())
        print("\nMissing Values per Column:")
        print(df_train.isnull().sum())

# Plot the distribution of 'SalePrice'
def plot_distribution(df_train, column='SalePrice'):
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.hist(df_train[column], bins=30, color='blue', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

# Plot correlation between numerical features and target ('SalePrice')
def plot_correlation(df_train):
    numeric_features = df_train.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_features.corr()
    correlation_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
    if verbose:
        plt.figure(figsize=(10, 8))
        correlation_with_target.drop('SalePrice').plot(kind='bar', color='red')
        plt.title('Correlation with SalePrice')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.show()
    return correlation_with_target

# Apply log transformation to skewed features
def apply_log_transformation(df_train, target_column):
    df_train[target_column] = np.log(df_train[target_column] + 1)
    return df_train

# Function to calculate skewness of each numerical feature
def calculate_skewness(df_train):
    skewness = df_train.select_dtypes(include=['int64', 'float64']).skew()
    return skewness

# Function to handle transformation based on skewness for continuous features
def transform_based_on_skewness(df_train):
    skewness = calculate_skewness(df_train)
    scaler = StandardScaler()
    
    # Only apply transformations to continuous features
    for col in features['Continuous']:
        skew = skewness[col]
        
        # If skewness is greater than 0.5, log transform
        if skew > 0.5:
            df_train[col] = np.log(df_train[col] + 1)
            print(f"Log transformation applied to {col} (right-skewed)")
        
        # If skewness is less than -0.5, square root transform (left-skewed)
        elif skew < -0.5:
            df_train[col] = np.sqrt(df_train[col])
            print(f"Square root transformation applied to {col} (left-skewed)")
        
        # If skewness is between -0.5 and 0.5, standardize (apply StandardScaler)
        else:
            df_train[col] = scaler.fit_transform(df_train[[col]])
            print(f"StandardScaler applied to {col} (normal or near-normal)")
    
    return df_train

# One-hot encoding for categorical features only
def one_hot_encode(df_train):
    df_train = pd.get_dummies(df_train, columns=features['Categorical'], drop_first=True)
    print("One-Hot Encoding applied to categorical features")
    return df_train

# Function to standardize continuous features
def standardize_continuous_features(df_train, continuous_features):
    scaler = StandardScaler()
    df_train[continuous_features] = scaler.fit_transform(df_train[continuous_features])
    print(f"Standardization applied to the continuous features: {continuous_features}")
    return df_train

# Function to split data into training and testing sets
def split_data(df_train, target_column='SalePrice'):
    X = df_train.drop(target_column, axis=1)  # Features
    y = df_train[target_column]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train and evaluate Linear Regression model with cross-validation
def train_and_evaluate_linear_regression(X_train, y_train):
    print(f"Linear Regression \n")
    
    model = LinearRegression()
    
    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    
    # Since cross_val_score returns negative MSE, we need to take the negative of it to get positive values
    cv_scores = -cv_scores
    
    # Calculate the mean and standard deviation of the cross-validation scores
    mean_mse = cv_scores.mean()
    std_mse = cv_scores.std()
    
    # Print results
    print(f"Cross-Validation Mean Squared Error (MSE): {mean_mse}")
    print(f"Cross-Validation Standard Deviation (MSE): {std_mse}")
    
    # Fit the model on the full training set and evaluate on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    
    # Calculate R-squared on the training set
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    
    print(f"Training R-squared: {r2}")
    print(f"Training RMSE: {rmse}")
    print(f"Training MAE: {mae}")
    if verbose:
        # Plot the predicted vs true values on training data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_train, y=y_pred)
        plt.title("True vs Predicted SalePrice (Linear regression Data)")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.show()

def train_and_evaluate_Random_forest(X_train, y_train, run_gridsearch=False):
    print("Random Forest")
    
    if run_gridsearch:
        print("Running GridSearchCV for hyperparameter tuning...")
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Hyperparameters for Random Forest:", best_params)
        model = grid_search.best_estimator_
    else:
        print("Skipping GridSearchCV. Using predetermined best Random Forest parameters.")
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores = -cv_scores
    mean_mse = cv_scores.mean()
    std_mse = cv_scores.std()
    print(f"Cross-Validation Mean Squared Error (MSE): {mean_mse}")
    print(f"Cross-Validation Standard Deviation (MSE): {std_mse}")


    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    
    print(f"Training R-squared: {r2}")
    print(f"Training RMSE: {rmse}")
    print(f"Training MAE: {mae}")
   
    if verbose:
        # Plot the predicted vs true values on training data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_train, y=y_pred)
        plt.title("True vs Predicted SalePrice (Random forest regression Data)")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.show()

# Function to remove outliers
def remove_outliers(df):
    for col, threshold in conditions.items():
        df = df[df[col] <= threshold]
    return df

# Feature engineering function
def feature_engineering(all_data):
    all_data["TotalSqrtFeet"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
    all_data["TotalBaths"] = all_data["BsmtFullBath"] + (all_data["BsmtHalfBath"] * 0.5) + all_data["FullBath"] + (all_data["HalfBath"] * 0.5)
    all_data['Isgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    all_data['Isfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    all_data['Ispool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    all_data['Issecondfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['IsOpenPorch'] = all_data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['IsWoodDeck'] = all_data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    return all_data


# Main function to execute the entire workflow
def main():
    # Step 1: Read the training data
    df_train = pd.read_csv('train.csv')
    df_train.drop('Id', axis=1, inplace=True)
    
    # Step 2: Apply the NaN value transformation
    df_train = none_transform(df_train)
    
    # Step 2: Apply the NaN value transformation
    df_train = remove_outliers(df_train)
    
    # Feature engineering
    df_train = feature_engineering(df_train)
    
    # Step 3: Explore data
    explore_data(df_train)

    # Step 4: Apply transformations based on skewness
    df_train = transform_based_on_skewness(df_train)

    # Step 5: One-hot encode categorical features
    df_train = one_hot_encode(df_train)

    # Step 6: Standardize continuous features
    continuous_features = features['Continuous']
    df_train = standardize_continuous_features(df_train, continuous_features)

    # Step 7: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_train)

    # Step 8: Train and evaluate the Linear Regression model with cross-validation
    train_and_evaluate_linear_regression(X_train, y_train)

    # Step 8: Train and evaluate the Linear Regression model with cross-validation
    train_and_evaluate_Random_forest(X_train, y_train)

# Run the main function
if __name__ == '__main__':
    main()
