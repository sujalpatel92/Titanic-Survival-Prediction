import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive preprocessing and feature engineering on the Titanic dataset.

    Args:
        df (pd.DataFrame): The raw Titanic DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model training.
    """
    data = df.copy()

    # 1. Handle Missing Values
    # Impute Age with median
    median_age = data['Age'].median()
    data['Age'].fillna(median_age, inplace=True)

    # Impute Embarked with mode
    mode_embarked = data['Embarked'].mode()[0]
    data['Embarked'].fillna(mode_embarked, inplace=True)

    # Handle Cabin: Create 'Has_Cabin' and drop original 'Cabin'
    data['Has_Cabin'] = data['Cabin'].notna().astype(int)
    data.drop('Cabin', axis=1, inplace=True)

    # 2. Feature Engineering
    # FamilySize and IsAlone
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # Title Extraction
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme'],
                                          ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Miss', 'Miss', 'Mrs'])
    data.drop('Name', axis=1, inplace=True)

    # Fare_Bin (Binning Fare)
    data['Fare_Bin'] = pd.qcut(data['Fare'], q=4, labels=['Very_Low', 'Low', 'Medium', 'High'])

    # Age_Bin (Binning Age)
    age_bins = [0, 12, 25, 60, 100]
    age_labels = ['Child', 'YoungAdult', 'Adult', 'Senior']
    data['Age_Bin'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

    # Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Ticket', 'Age', 'Fare', 'SibSp', 'Parch']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    # 3. Encoding Categorical Features and Scaling Numerical Features
    # Identify categorical and numerical columns for transformation
    # Pclass is numerical but treated as categorical due to its discrete nature
    categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title', 'Fare_Bin', 'Age_Bin']

    # For this function, we'll assume Age and Fare have been binned and removed,
    # and FamilySize and IsAlone are the primary numerical features (already created).
    # If there were other numerical features *not* binned, they'd go here.
    # Given the notebook dropped original Age/Fare, we focus on the encoded and new features.
    
    # For a direct translation of the notebook, all relevant features should be processed.
    # The notebook ultimately kept Age_Bin, Fare_Bin (categorical) and scaled FamilySize.
    # We will use ColumnTransformer to ensure consistent processing.

    # Features that are truly numerical and need scaling
    # (after dropping original Age/Fare and keeping FamilySize)
    numerical_features_to_scale = [] # Since Age and Fare are binned, and SibSp/Parch are part of FamilySize

    # Features that are categorical and need one-hot encoding
    # These include the newly created binned features as well
    categorical_features_to_encode = [col for col in data.columns if data[col].dtype == 'object' or col in ['Has_Cabin', 'IsAlone', 'Pclass', 'Fare_Bin', 'Age_Bin']]
    
    # Create preprocessing pipelines for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            # ('num', StandardScaler(), numerical_features_to_scale), # If we had continuous numerical features to scale
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_to_encode)
        ],
        remainder='passthrough' # Keep other columns (like 'Survived')
    )
    
    # Separate target variable before transformation
    if 'Survived' in data.columns:
        y = data['Survived']
        X = data.drop('Survived', axis=1)
    else:
        y = None # Or handle test data case
        X = data

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_to_encode)
    
    # Combine numerical (if any) and encoded categorical feature names
    # Note: If remainder='passthrough', those column names also need to be retrieved.
    # For simplicity, we manually combine based on what's left after dropping initial columns.
    
    # This part can be tricky without a full pipeline that tracks all column names.
    # For this exercise, let's assume the final columns are mostly the encoded ones.
    # We will reconstruct the DataFrame by combining the encoded features and the
    # 'Survived' column (if it exists).

    # Get the names of the columns that were passed through (e.g., 'Has_Cabin', 'IsAlone') if not already encoded
    # The ColumnTransformer will process 'Has_Cabin' and 'IsAlone' if they are listed in categorical_features_to_encode
    # If they were numerical and we had a numerical transformer, they'd be handled there.
    # Given their 0/1 nature, treating them as categorical for OneHotEncoder is also valid.
    
    # Reconstruct DataFrame with proper column names
    # First, get the list of all transformed column names
    all_transformed_features = list(encoded_feature_names) # Assuming all features go through 'cat' for simplicity

    # If 'Has_Cabin' and 'IsAlone' were not explicitly passed to one-hot encoder due to being binary already,
    # they would be in 'remainder'. Let's ensure they are handled properly by including them in categorical_features_to_encode.
    # After the `preprocessor.fit_transform(X)`, `X_processed` will contain the new features.
    
    # A more robust way to get all column names after ColumnTransformer:
    new_column_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            new_column_names.extend(transformer.get_feature_names_out(cols))
        elif name == 'remainder':
            # Identify columns that were passed through
            remaining_cols_idx = preprocessor.transformers_[0][2].shape[0] if len(preprocessor.transformers_[0][2]) > 0 else 0
            # This is a simplification; robustly identifying remainder columns by name is complex
            # and might require tracking input column indices to the ColumnTransformer.
            # For this skeleton, we assume all processed features are covered by `get_feature_names_out`
            # or are manually added if they were true remainders like a numerical feature.

    # Simpler approach: build the final DataFrame after encoding and before saving.
    # For 'preprocess.py', we want to return a DataFrame that looks like what we saved in the notebook.
    # Let's re-run the transformation steps manually to control column names and ordering.

    # Re-initialise data from df.copy() to apply transformations step-by-step
    data = df.copy()
    median_age = data['Age'].median()
    data['Age'].fillna(median_age, inplace=True)
    mode_embarked = data['Embarked'].mode()[0]
    data['Embarked'].fillna(mode_embarked, inplace=True)
    data['Has_Cabin'] = data['Cabin'].notna().astype(int)
    data.drop('Cabin', axis=1, inplace=True)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme'],
                                          ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Miss', 'Miss', 'Mrs'])
    data.drop('Name', axis=1, inplace=True)
    data['Fare_Bin'] = pd.qcut(data['Fare'], q=4, labels=['Very_Low', 'Low', 'Medium', 'High'])
    age_bins = [0, 12, 25, 60, 100]
    age_labels = ['Child', 'YoungAdult', 'Adult', 'Senior']
    data['Age_Bin'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

    # Drop columns that won't be used or are replaced
    columns_to_drop_final = ['PassengerId', 'Ticket', 'Age', 'Fare', 'SibSp', 'Parch']
    data.drop(columns=[col for col in columns_to_drop_final if col in data.columns], inplace=True)

    # Identify numerical and categorical features for final processing
    # 'Has_Cabin' and 'IsAlone' are already 0/1 integers, good to go
    # 'Survived' is the target
    target_column = 'Survived' if 'Survived' in data.columns else None
    
    numerical_cols = [] # As per the notebook, Age/Fare are binned, FamilySize is not scaled here as it's not a continuous original feature
                        # but a derived count. If scaled, it needs a scaler. For consistency with notebook, if it was scaled, scale here.
    
    # In the notebook, FamilySize was scaled as a numerical_cols_for_scaling. Let's include it.
    if 'FamilySize' in data.columns:
        numerical_cols.append('FamilySize')

    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' or col in ['Pclass', 'Fare_Bin', 'Age_Bin']]
    # We explicitly exclude 'Has_Cabin' and 'IsAlone' from one-hot encoding if they are already binary ints
    # and we want to keep them as is. If we want to one-hot encode them, include them in categorical_cols.
    # The notebook did include Pclass, Fare_Bin, Age_Bin along with Sex, Embarked, Title for OneHotEncoding.
    # Let's ensure 'Has_Cabin' and 'IsAlone' are handled correctly (e.g., if they are also considered categorical)
    # They are already numerical 0/1, so they can be treated as numerical and potentially scaled, or kept as is.
    # For now, let's include them in the categorical_cols to be one-hot encoded for consistency with the notebook's approach
    # if it results in separate binary features like 'Has_Cabin_0', 'Has_Cabin_1'.
    # However, 'Has_Cabin' and 'IsAlone' are typically kept as single binary features.
    # Let's adjust to only one-hot encode truly multi-category features and scale remaining numerical.

    features_for_ohe = [col for col in categorical_cols if data[col].nunique() > 2] # Multi-category
    features_for_binary_ohe = [col for col in categorical_cols if data[col].nunique() == 2] # Binary-category (can be kept as is or OHE)

    # In the notebook, Sex, Embarked, Pclass, Title, Fare_Bin, Age_Bin were all one-hot encoded.
    # Let's stick to that for this function.
    categorical_features_for_pipeline = ['Sex', 'Embarked', 'Pclass', 'Title', 'Fare_Bin', 'Age_Bin']
    
    # Numerical features that should be scaled (FamilySize was scaled in notebook)
    numerical_features_for_pipeline = []
    if 'FamilySize' in data.columns and data['FamilySize'].dtype != 'object':
        numerical_features_for_pipeline.append('FamilySize')
    
    # Create the preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_for_pipeline),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_for_pipeline)
        ],
        remainder='passthrough' # Keep 'Has_Cabin', 'IsAlone' and 'Survived' if they are not processed
    )

    # Separate X and y before fitting preprocessor
    X = data.drop(columns=[target_column]) if target_column else data
    
    # Fit and transform X
    X_transformed = preprocessor.fit_transform(X)

    # Get names of new columns
    transformed_column_names = []
    for name, transformer, original_cols in preprocessor.transformers_:
        if name == 'cat':
            transformed_column_names.extend(transformer.get_feature_names_out(original_cols))
        elif name == 'num':
            transformed_column_names.extend(original_cols) # Scaled numerical columns keep original names
        elif name == 'remainder':
            # This captures columns not explicitly handled.
            # In our case, 'Has_Cabin' and 'IsAlone' are binary ints, so they might be here if not explicitly put into 'num' or 'cat'.
            # Given they are int, they won't be in categorical_features_for_pipeline.
            # Let's ensure 'Has_Cabin' and 'IsAlone' are explicitly handled, either by numerical scaling or by leaving them as is.
            # In the notebook, they were implicitly handled by not being dropped and not specifically scaled/encoded,
            # indicating they were expected to remain as numerical 0/1.

            # Re-evaluating based on notebook's output:
            # The notebook's final `data.head()` after all steps shows 'Has_Cabin', 'IsAlone',
            # and the one-hot encoded features. It also shows `Age` and `Fare` dropped.
            # `FamilySize` was scaled.
            # So, the numerical features to scale should be ['Age', 'Fare', 'FamilySize']
            # when they are present. But Age/Fare were dropped, so only FamilySize.
            # 'Has_Cabin', 'IsAlone' remained as features.

            # Let's simplify the preprocessor for the script to match the notebook's final outcome more closely:
            # We already handled Age, Fare, Cabin, Name, SibSp, Parch.
            # The remaining features are 'Survived', 'Has_Cabin', 'IsAlone', and the binned/title features.

            # Re-structuring the pipeline for cleaner feature handling in the script
            # 1. Start with `data` after initial missing value and simple feature engineering
            # (FamilySize, IsAlone, Has_Cabin, Title, Fare_Bin, Age_Bin).
            # 2. Define numerical and categorical features for the final sklearn pipeline.

            # Features that are now numerical (after deriving and initial drops)
            numerical_features_final = ['FamilySize'] # This is the only continuous-like numerical feature we have left to scale
            # 'Has_Cabin' and 'IsAlone' are already 0/1, no need to scale or one-hot encode further if they remain binary.
            # If we were to one-hot encode Has_Cabin/IsAlone, they would become two columns each (e.g., Has_Cabin_0, Has_Cabin_1).
            # The notebook leaves them as single 0/1 columns which is more efficient.

            # Features that are categorical and need one-hot encoding
            categorical_features_final = ['Sex', 'Embarked', 'Pclass', 'Title', 'Fare_Bin', 'Age_Bin']
            
            # Create a preprocessing pipeline for these specific features
            final_preprocessor = ColumnTransformer(
                transformers=[
                    ('num_scaler', StandardScaler(), numerical_features_final),
                    ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_final)
                ],
                remainder='passthrough' # This will pass through 'Has_Cabin', 'IsAlone' and 'Survived' (if present)
            )

            # Separate features (X) and target (y)
            X_temp = data.drop(columns=[target_column]) if target_column else data
            
            # Apply transformation
            X_transformed_array = final_preprocessor.fit_transform(X_temp)

            # Get feature names for the transformed array
            feature_names = []
            feature_names.extend(numerical_features_final) # Scaled numerical features
            feature_names.extend(final_preprocessor.named_transformers_['cat_encoder'].get_feature_names_out(categorical_features_final)) # Encoded categorical features
            
            # Add remainder column names (Has_Cabin, IsAlone)
            # This is tricky with ColumnTransformer if not explicitly listed.
            # A more robust way to get remainder names is often to inspect the original columns
            # that were not part of any transformer.
            remaining_cols = [col for col in X_temp.columns if col not in numerical_features_final + categorical_features_final]
            feature_names.extend(remaining_cols) # This will add Has_Cabin and IsAlone

            processed_data = pd.DataFrame(X_transformed_array, columns=feature_names, index=data.index)
            
            # Re-add the target column if it existed
            if target_column:
                processed_data[target_column] = y

            return processed_data


if __name__ == '__main__':
    # This block is for testing the function directly
    print("Running preprocess.py as a script for testing...")
    # Create a dummy DataFrame that mimics the structure of train.csv
    dummy_data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 1, 0],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, None],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
        'Cabin': [None, 'C85', None, 'C123', None],
        'Embarked': ['S', 'C', 'S', 'S', 'Q']
    }
    dummy_df = pd.DataFrame(dummy_data)

    print("\\nOriginal Dummy DataFrame:")
    print(dummy_df.info())
    print(dummy_df.head())

    processed_dummy_df = preprocess_titanic(dummy_df)

    print("\\nProcessed Dummy DataFrame:")
    print(processed_dummy_df.info())
    print(processed_dummy_df.head())
    print(processed_dummy_df.columns)