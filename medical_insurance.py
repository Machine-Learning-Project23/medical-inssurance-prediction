import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas DataFrame
df=pd.read_csv("Medical_insurance.csv")

print("\n")
# Display data information
print("<<<<<Display data information>>>>>"+"\n")
info=df.info()
print(info,"\n")

df["region"].unique()
df.head()

df['sex']=df['sex'].replace("female",1)
df['sex']=df['sex'].replace("male",0)
df['smoker']=df['smoker'].replace("yes",1)
df['smoker']=df['smoker'].replace("no",0)

ohe =OneHotEncoder()
feature_arry = ohe.fit_transform(df[["region"]]).toarray()
feature_labels = ohe.categories_
feature_labels=list(chain.from_iterable(feature_labels))
feature_labels[:]=['southwest', 'southeast','northwest','northeast']
features = pd.DataFrame(feature_arry, columns = feature_labels)
df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
df = df.drop(['region'],  axis=1)

df['bmi']=df['bmi'].astype(int)
df['southeast']=df['southeast'].astype(int)
df['southwest']=df['southwest'].astype(int)
df['northeast']=df['northeast'].astype(int)
df['northwest']=df['northwest'].astype(int)
df['charges']=df['charges'].astype(int)
df.info()

# Split the dataset into features (X) and target variable (charges)
X = df.drop(['charges'], axis=1)
charges = df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, charges, test_size=0.3, random_state=42)

X.head()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the regressors
logistic_regression = LogisticRegression(max_iter=2, random_state=42)
logistic_regression.fit(X_train, y_train)
random_forest = RandomForestRegressor(n_estimators=100, random_state=1)
random_forest.fit(X_train, y_train)
decision_tree = DecisionTreeRegressor(max_depth=3)
decision_tree.fit(X_train, y_train)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)
svm = svm.SVR(kernel='linear')
svm.fit(X_train, y_train)
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

logistic_regression_predictions = logistic_regression.predict(X_test)
random_forest_predictions = random_forest.predict(X_test)
decision_tree_predictions = decision_tree.predict(X_test)
knn_predictions = knn.predict(X_test)
svm_predictions = svm.predict(X_test)
naive_bayes_predictions = naive_bayes.predict(X_test)

# Calculate mean squared error
logistic_regression_mse = mean_squared_error(y_test, logistic_regression_predictions)
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
knn_mse = mean_squared_error(y_test, knn_predictions)
svm_mse = mean_squared_error(y_test, svm_predictions)
naive_bayes_mse = mean_squared_error(y_test, naive_bayes_predictions)

print("logistic_regression_mse: ",logistic_regression_mse)
print("random_forest_mse: ",random_forest_mse)
print("decision_tree_mse: ",decision_tree_mse)
print("knn_mse: ",knn_mse)
print("svm_mse: ",svm_mse)
print("naive_bayes_mse: ",naive_bayes_mse)

# Calculate root mean square error
logistic_regression_rmse = np.sqrt(logistic_regression_mse)
random_forest_rmse = np.sqrt(random_forest_mse)
decision_tree_rmse = np.sqrt(decision_tree_mse)
knn_rmse = np.sqrt(knn_mse)
svm_rmse = np.sqrt(svm_mse)
naive_bayes_rmse = np.sqrt(naive_bayes_mse)

print("logistic_regression_rmse: ",logistic_regression_rmse)
print("random_forest_rmse: ",random_forest_rmse)
print("decision_tree_rmse: ",decision_tree_rmse)
print("knn_rmse: ",knn_rmse)
print("svm_rmse: ",svm_rmse)
print("naive_bayes_mse: ",naive_bayes_mse)

# Calculate R-squared
logistic_regression_r2 = r2_score(y_test, logistic_regression_predictions)
random_forest_r2 = r2_score(y_test, random_forest_predictions)
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)
knn_r2 = r2_score(y_test, knn_predictions)
svm_r2 = r2_score(y_test, svm_predictions)
naive_bayes_r2 = r2_score(y_test, naive_bayes_predictions)

print("logistic_regression_r2: ",logistic_regression_r2)
print("random_forest_r2: ",random_forest_r2)
print("decision_tree_r2: ",decision_tree_r2)
print("knn_r2: ",knn_r2)
print("svm_r2: ",svm_r2)
print("naive_bayes_r2: ",naive_bayes_r2)

# Calculate mean absolute error
logistic_regression_mae = mean_absolute_error(y_test, logistic_regression_predictions)
random_forest_mae = mean_absolute_error(y_test, random_forest_predictions)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)
svm_mae = mean_absolute_error(y_test, svm_predictions)
naive_bayes_mae = mean_absolute_error(y_test, naive_bayes_predictions)

print("logistic_regression_mae: ",logistic_regression_mae)
print("random_forest_mae: ",random_forest_mae)
print("decision_tree_mae: ",decision_tree_mae)
print("knn_mae: ",knn_mae)
print("svm_mae: ",svm_mae)
print("naive_bayes_mae: ",naive_bayes_mae)

# The mean absolute percentage error
logistic_regression = 100 * (np.abs(y_test - logistic_regression_predictions) / y_test)
logistic_regression_mape=np.mean(logistic_regression)

random_forest = 100 * (np.abs(y_test - random_forest_predictions) / y_test)
random_forest_mape=np.mean(random_forest)

decision_tree = 100 * (np.abs(y_test - decision_tree_predictions) / y_test)
decision_tree_mape =np.mean(decision_tree)

knn = 100 * (np.abs(y_test - knn_predictions) / y_test)
knn_mape=np.mean(knn)

svm = 100 * (np.abs(y_test - svm_predictions) / y_test)
svm_mape = np.mean(svm)

naive_bayes=100 * (np.abs(y_test - naive_bayes_predictions) / y_test)
naive_bayes_mape =np.mean(naive_bayes)

print('logistic_regression_MAPE: ',logistic_regression_mape)
print('random_forest_MAPE: ', random_forest_mape)
print('decision_tree_MAPE: ', decision_tree_mape)
print('KNN_MAPE: ', knn_mape)
print('SVM_MAPE: ', svm_mape)
print('naive_bayes_MAPE: ', naive_bayes_mape)

# Create a DataFrame to store the results
results = pd.DataFrame({'True Labels': y_test,
                        'Logistic Regression Predictions': logistic_regression_predictions,
                        'Random Forest Predictions': random_forest_predictions,
                        'Decision Tree Predictions': decision_tree_predictions,
                        'KNN Predictions': knn_predictions,
                        'SVM Predictions': svm_predictions,
                        'Naive Bayes Predictions': naive_bayes_predictions
                        })

styled_results = results.style.background_gradient(cmap='coolwarm')
r = pd.DataFrame(results)
print("\n")
print("<<<<<Print the prediction of the true labels and each algorithm>>>>>"+"\n")

formatted_df = tabulate(
    results,
    headers=results.columns,
    tablefmt="fancy_grid",
)
print(formatted_df)

# Create a DataFrame to store the results
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree','KNN','SVM','Naive Bayes'],
    'Mean Square Error': [logistic_regression_mse, random_forest_mse,decision_tree_mse,knn_mse,svm_mse,naive_bayes_mse],
    'Root Mean Square Error': [logistic_regression_rmse,random_forest_rmse,decision_tree_rmse,knn_rmse,svm_rmse,naive_bayes_rmse],
    'R-squared': [logistic_regression_r2,random_forest_r2,decision_tree_r2,knn_r2,svm_r2,naive_bayes_r2],
    'Mean Absolute Error': [logistic_regression_mae,random_forest_mae,decision_tree_mae,knn_mae,svm_mae,naive_bayes_mae],
    'Mean Absolute Percentage Error': [logistic_regression_mape,random_forest_mape,decision_tree_mape,knn_mape,svm_mape,naive_bayes_mape]
})

# Print the prediction results
print("<<<<<Prediction Results>>>>>")
formatted_df = tabulate(
    results,
    headers=results.columns,
    tablefmt="fancy_grid",
)
print(formatted_df)

r2 = [logistic_regression_r2,random_forest_r2,decision_tree_r2,knn_r2,svm_r2,naive_bayes_r2]
mse=[logistic_regression_mse, random_forest_mse,decision_tree_mse,knn_mse,svm_mse,naive_bayes_mse]
rmse= [logistic_regression_rmse,random_forest_rmse,decision_tree_rmse,knn_rmse,svm_rmse,naive_bayes_rmse]
mae=[logistic_regression_mae,random_forest_mae,decision_tree_mae,knn_mae,svm_mae,naive_bayes_mae]
mape=[logistic_regression_mape,random_forest_mape,decision_tree_mape,knn_mape,svm_mape,naive_bayes_mape]

print(r2.index(max(r2)))
print(mse.index(min(mse)))
print(rmse.index(min(rmse)))
print(mae.index(min(mae)))
print(mape.index(min(mape)))

Mean_Square_Error_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree','KNN','SVM','Naive Bayes'],
    'Mean Square Error': [logistic_regression_mse, random_forest_mse,decision_tree_mse,knn_mse,svm_mse,naive_bayes_mse],
})

plt.figure(figsize=(8, 7))
plt.bar(Mean_Square_Error_df['Model'], Mean_Square_Error_df['Mean Square Error'])
plt.xlabel('Regressor')
plt.ylabel('Mean Square Error')
plt.title ('Mean Square Error of Different Regressors')
plt.ylim(1000, 9000)
plt.xticks(rotation=45)
plt.show()


Root_Mean_Square_Error_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree','KNN','SVM','Naive Bayes'],
    'Root Mean Square Error': [svm_rmse, naive_bayes_rmse, logistic_regression_rmse,random_forest_rmse, knn_rmse, decision_tree_rmse]
})

plt.figure(figsize=(8, 7))
plt.bar(Root_Mean_Square_Error_df['Model'], Root_Mean_Square_Error_df['Root Mean Square Error'])
plt.xlabel('Regressor')
plt.ylabel('Root Mean Square Error')
plt.title ('Root Mean Square Error of Different Regressors')
plt.ylim(1000, 9000)
plt.xticks(rotation=45)
plt.show()


R_squared = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree','KNN','SVM','Naive Bayes'],
    'R-squared': [logistic_regression_r2,random_forest_r2,decision_tree_r2,knn_r2,svm_r2,naive_bayes_r2],
})

plt.figure(figsize=(8, 7))
plt.bar(R_squared['Model'], R_squared['R-squared'])
plt.xlabel('Regressor')
plt.ylabel('R-Squared')
plt.title ('R-Squared of Different Regressors')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()



mae = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree','KNN','SVM','Naive Bayes'],
    'Mean Absolute Error': [logistic_regression_mae,random_forest_mae,decision_tree_mae,knn_mae,svm_mae,naive_bayes_mae]
})

plt.figure(figsize=(8, 7))
plt.bar(mae['Model'], mae['Mean Absolute Error'])
plt.xlabel('Regressor')
plt.ylabel('Mean Absolute Error')
plt.title ('Mean Absolute Error of Different Regressors')
plt.ylim(1000,9000)
plt.xticks(rotation=45)
plt.show()
