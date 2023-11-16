import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def model_data_scikit(df: pd.DataFrame, numeric_features: list, categorical_features: list, y_field: str):
    # # Split the data into features and target variable
    all_features = numeric_features + categorical_features
    X = df[all_features]
    y = df[y_field]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create a pipeline with the preprocessor and linear regression model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def model_data_stats_model(df: pd.DataFrame, numeric_features: list, categorical_features: list, y_field: str):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # Encode categorical feature using OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categoricals = []
    for categorical_feature in categorical_features:
        unique_fields = sorted(df[categorical_feature].unique())
        encoded_categorical = pd.DataFrame(encoder.fit_transform(df[categorical_feature].values.reshape(-1,1)),
                                           columns=unique_fields[1:])
        encoded_categoricals.append(encoded_categorical)
    df_categoricals = pd.concat(encoded_categoricals, axis=1)
    df_numericals = df[numeric_features]
    # Concatenate numerical and encoded categorical features
    X = pd.concat([df_numericals, df_categoricals], axis=1)

    # Add a constant column for the intercept term
    X = sm.add_constant(X)

    y = df[y_field]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model using statsmodels
    model = sm.OLS(y_train, X_train).fit()

    # Print the summary of the regression
    print(model.summary())

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    print(f'Mean Squared Error: {mse}')

    print("here")