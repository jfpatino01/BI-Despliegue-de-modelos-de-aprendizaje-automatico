import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# define the function to clip the values
numerical_columns1 = [0]
numerical_columns2 = [1]
categorical_columns = ['owner', 'seller_type', 'fuel']
scale_columns = [12, 13]
features = ['year', 'mileage', 'owner', 'seller_type','fuel',]

def clip_values(X, lower_bound, upper_bound):
    X_clipped = np.clip(X, lower_bound, upper_bound)
    return X_clipped


clip_transformer1 = FunctionTransformer(clip_values, kw_args={'lower_bound': 1994, 'upper_bound': 2020})
clip_transformer2 = FunctionTransformer(clip_values, kw_args={'lower_bound': 0, 'upper_bound': 46.816})

pipeline = Pipeline(
    [
        ('initial',ColumnTransformer(transformers=[('passthrough', 'passthrough', features)])),
        ('validRange',  ColumnTransformer(transformers=[('clip', clip_transformer1, numerical_columns1)], remainder='passthrough')),
        ('validRange2',  ColumnTransformer(transformers=[('clip', clip_transformer2, numerical_columns2)], remainder='passthrough')),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('dummyOwner', ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [3])], remainder='passthrough')),
        ('dummySellerType', ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [5])], remainder='passthrough')),
        ('dummyFuel', ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [10])], remainder='passthrough')),
        ('scaler', ColumnTransformer(transformers=[('scaler', StandardScaler(), scale_columns)], remainder='passthrough')),
        ('model', LinearRegression())
    ]
)