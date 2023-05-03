########################################################################################################
# screening test
# Import necessary libraries
########################################################################################################
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor as GBTRegressor
)
from sklearn.ensemble import GradientBoostingClassifier as GBTClassifier
from sklearn.linear_model import Ridge as RidgeRegression
from sklearn.linear_model import Lasso as LassoRegression
from sklearn.linear_model import ElasticNet as ElasticNetRegression
from sklearn import svm as SVM
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import ExtraTreesClassifier as extra_random_trees
from sklearn.neural_network import MLPClassifier as neural_network
#!pip install xgboost
import xgboost as xgb
from xgboost import XGBRegressor

########################################################################################################
# Finding error in json file
########################################################################################################
import json

with open("algo_from_ui_modif.json") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

########################################################################################################
# Load data from CSV file using pandas
########################################################################################################
df = pd.read_csv("iris.csv")

# Load algorithm parameters from JSON file
with open("algo_from_ui_modif.json") as f:
    algoparams = json.load(f)
    
########################################################################################################
# Read the target and type of regression to be run
########################################################################################################
algoparams["target"]["prediction_type"]["Target"] = input("\n Enter target:")
algoparams["target"]["prediction_type"]["type"] = input("\n Enter prediction_type:")
algoparams["target"]["prediction_type"]["model_type"] = input("\n Enter model_type:")
# Save the updated dictionary back to the JSON file
with open("algoparams.json", "w") as f:
    json.dump(algoparams, f, indent=4)

########################################################################################################
# Test Scenario:
########################################################################################################
algoparams["target"]["feature_handling"][
    algoparams["target"]["prediction_type"]["Target"]
]

########################################################################################################
# Test Scenario:
########################################################################################################
algoparams["target"]["algorithms"][
    algoparams["target"]["prediction_type"]["model_type"]
]

########################################################################################################
# Read the features (which are column names in the csv) and figure out what missing imputation needs
# to be applied and apply that to the columns loaded in a dataframe
########################################################################################################
for outer_key, outer_value in algoparams["target"]["feature_handling"].items():
    print("########################################################")
    for inner_key, inner_value in outer_value.items():
        try:
            val_imput = outer_value["feature_details"]["impute_with"]
        except KeyError as K:
            pass
        # for inner_key_2, inner_value_2 in value.items():
        print(inner_key, inner_value)
        
        if inner_key == "feature_details":
            # val_imput= outer_value["feature_details"]["impute_with"]
            if val_imput == "Average of values" and outer_key == "sepal_length":
                print("Error Average of values")
                # df = df.fillna(0)
                df[outer_key].fillna(0, inplace=True)
            elif val_imput == "Average of values" and outer_key == "petal_length":
                print("Error Average of values")
                # df = df.fillna(0)
                df[outer_key].fillna(0, inplace=True)
            if val_imput == "custom":
                if outer_key == "petal_width":
                    print("Error petal_width")
                    df[outer_key].fillna(-2, inplace=True)
                    # df = df.fillna(-2)
                if outer_key == "sepal_width":
                    print("Error sepal_width")
                    df[outer_key].fillna(-1, inplace=True)
                    
########################################################################################################
# copying data frame in another data frame
########################################################################################################
dataframe = df.copy()
df2 = df.copy()
v_num_of_feature = int(
    algoparams["target"]["feature_reduction"]["num_of_features_to_keep"]
)

########################################################################################################
# Compute feature reduction based on input. See the screenshot below where there can be No Reduction,
# Corr with Target, Tree-based, PCA. Please make sure you write code so that all options can work.
# If we rerun your code with a different Json it should work if we switch No Reduction to say PCA.
########################################################################################################
dataframe = dataframe.drop("species", axis=1)

# Get the feature reduction method from the JSON parameters
# reduction_method = algoparams.get("feature_reduction", "No Reduction")
reduction_method = algoparams["target"]["feature_reduction"]["feature_reduction_method"]
target_variable_fr = algoparams["target"]["prediction_type"]["Target"]
# Split data into features (X) and target (y)
X = dataframe.drop(target_variable_fr, axis=1)
y = dataframe[target_variable_fr]

def compute_feature_reduction(dataframe, reduction_option):
    if reduction_method == "No Reduction":
        # No feature reduction, use all features
        X_reduced = X
        return X_reduced
    elif reduction_method == "Corr with Target":
        # Compute correlation with target and keep top n features
        n_features = algoparams.get("n_features", v_num_of_feature)
        corr_with_target = X.corrwith(y)
        top_features = corr_with_target.abs().nlargest(n_features).index
        X_reduced = X[top_features]
        return X_reduced
    elif reduction_method == "Tree-based":
        # Use tree-based feature reduction for regression
        n_features = algoparams.get("n_features", v_num_of_feature)
        model = RandomForestRegressor()
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = feature_importances.nlargest(n_features).index
        X_reduced = X[top_features]
        return X_reduced
    elif reduction_method == "PCA":
        # Use PCA for feature reduction
        n_components = algoparams.get("n_components", v_num_of_feature)
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return X_reduced
    else:
        raise ValueError("Invalid feature reduction method")

########################################################################################################
# Test case:
########################################################################################################
# dataframe=dataframe.drop('species',axis=1)
dataframe_test = dataframe.rename(
    columns={algoparams["target"]["prediction_type"]["Target"]: "target"}
)

print(
    "Feature reduction type:",
    algoparams["target"]["feature_reduction"]["feature_reduction_method"],
)
# specify reduction option
reduction_option = algoparams["target"]["feature_reduction"]["feature_reduction_method"]

# compute feature reduction
reduced_dataframe = compute_feature_reduction(dataframe_test, reduction_option)

# display original and reduced dataframes
print("Original dataframe:\n", dataframe_test)
print("\nReduced dataframe:\n", reduced_dataframe)

########################################################################################################
# Parse the Json and make the model objects (using sklean) that can handle what is required
# in the “prediction_type” specified in the JSON (See #1 where “prediction_type” is specified).
# Keep in mind not to pick models that don’t apply for the prediction_type specified
# Model building for what is required in prediction type
########################################################################################################
from sklearn.metrics import mean_squared_error, r2_score

# Extract the prediction type from the inputted target
prediction_type = algoparams["target"]["prediction_type"]["type"]
# Extract the target variable column name from the algoparams
target_var = algoparams["target"]["prediction_type"]["Target"]

# Perform any necessary preprocessing or transformations based on the prediction type
if prediction_type == "classification":
    one_hot_encoded = pd.get_dummies(df)
elif prediction_type == "regression":
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled_data = scaler.transform(dataframe)
    scaled_df = pd.DataFrame(scaled_data, columns=dataframe.columns)
    X_sc = scaled_df.drop(target_var, axis=1)
    y_sc = scaled_df[target_var]
    if algoparams["target"]["train"]["policy"] == "Split the dataset":
        if algoparams["target"]["train"]["split"] == "Randomly":
            (
                X_train_scaled,
                X_test_scaled,
                y_train_scaled,
                y_test_scaled,
            ) = train_test_split(X_sc, y_sc, test_size=0.3, random_state=1)
            regression_model = LinearRegression()
            regression_model.fit(X_train_scaled, y_train_scaled)
            # Use the model to make predictions on the test data
            y_pred = regression_model.predict(X_test_scaled)
            y_pred
            # Evaluate the performance of the model using mean squared error
            mse = mean_squared_error(y_test_scaled, y_pred)
            print("Mean Squared Error:", mse)
            r2 = r2_score(y_test_scaled, y_pred)
            print("Root Mean Squared Error:", r2)
        else:
            pass

########################################################################################################
# JSON data with lowercase boolean values
########################################################################################################
json_data = '{"is_selected": false}'

# Load the JSON data with lowercase booleans converted to Python constants
data = json.loads(json_data, parse_constant=True)

# Print the loaded data
print(data)

for model_name, model_params in algoparams["target"]["algorithms"].items():
    param_grid = list(model_params.values())

########################################################################################################
#  Run the fit and predict on each model – keep in mind that you need to do hyper parameter tuning
# i.e., use GridSearchCV,Log to the console the standard model metrics that apply
########################################################################################################
# This is prepare for Regression model
# Define the pipelines with scaling and the models
pipelines = []
pipelines.append(
    ("LR", Pipeline([("scl", StandardScaler()), ("clf", LinearRegression())]))
)
pipelines.append(
    ("DT", Pipeline([("scl", StandardScaler()), ("clf", DecisionTreeRegressor())]))
)
pipelines.append(
    ("RF", Pipeline([("scl", StandardScaler()), ("clf", RandomForestRegressor())]))
)
pipelines.append(("SVM", Pipeline([("scl", StandardScaler()), ("clf", SVR())])))
pipelines.append(
    ("XGB", Pipeline([("scl", StandardScaler()), ("clf", XGBRegressor())]))
)

# Define the hyperparameters for grid search
hyperparameters = {
    "LR": {},
    "DT": {"clf__max_depth": [3, 5, 7, 9]},
    "RF": {"clf__n_estimators": [100, 200, 300], "clf__max_features": ["sqrt", "log2"]},
    "SVM": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__kernel": ["linear", "rbf"]},
    "XGB": {"clf__max_depth": [3, 5, 7, 9], "clf__n_estimators": [100, 200, 300]},
}

# Fit the models with Grid Search
results = []
names = []

for name, pipeline in pipelines:
    clf = GridSearchCV(pipeline, hyperparameters[name], cv=5, n_jobs=-1)
    clf.fit(X_train_scaled, y_train_scaled)
    y_pred = clf.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_scaled, y_pred)
    results.append(r2)
    names.append(name)
    print("{}: RMSE {:.4f}, R2 {:.4f}".format(name, rmse, r2))

# Select the best model
best_idx = np.argmax(results)
best_name = names[best_idx]
best_model = pipelines[best_idx][1]
print("Best Model: {}".format(best_name))

########################################################################################################
# This is prepare for Classification model
########################################################################################################
target_variable_fr = algoparams["target"]["prediction_type"]["Target"]
# Split data into features (X) and target (y)
X = df2.drop(target_variable_fr, axis=1)
y = df2[target_variable_fr]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the pipelines with scaling and the models
pipelines = []
pipelines.append(
    ("LR", Pipeline([("scl", StandardScaler()), ("clf", LogisticRegression())]))
)
pipelines.append(
    ("KNN", Pipeline([("scl", StandardScaler()), ("clf", KNeighborsClassifier())]))
)
pipelines.append(
    ("DT", Pipeline([("scl", StandardScaler()), ("clf", DecisionTreeClassifier())]))
)
pipelines.append(
    ("RF", Pipeline([("scl", StandardScaler()), ("clf", RandomForestClassifier())]))
)
pipelines.append(("SVM", Pipeline([("scl", StandardScaler()), ("clf", SVC())])))
pipelines.append(
    ("XGB", Pipeline([("scl", StandardScaler()), ("clf", XGBClassifier())]))
)

# Define the hyperparameters for grid search
hyperparameters = {
    "LR": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9]},
    "DT": {"clf__max_depth": [3, 5, 7, 9]},
    "RF": {"clf__n_estimators": [100, 200, 300], "clf__max_features": ["sqrt", "log2"]},
    "SVM": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__kernel": ["linear", "rbf"]},
    "XGB": {"clf__max_depth": [3, 5, 7, 9], "clf__n_estimators": [100, 200, 300]},
}

# Fit the models with Grid Search
results = []
names = []

for name, pipeline in pipelines:
    clf = GridSearchCV(pipeline, hyperparameters[name], cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(accuracy)
    names.append(name)
    print("{}: {:.4f}".format(name, accuracy))

# Select the best model
best_idx = np.argmax(results)
best_name = names[best_idx]
best_model = pipelines[best_idx][1]
print("Best Model: {}".format(best_name))

########################################################################################################
