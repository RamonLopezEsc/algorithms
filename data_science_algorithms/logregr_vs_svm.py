import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score,precision_score, recall_score


# %% Data QA Report
def dqr(data):
    col_names = list(data.columns.values)
    cols = pd.DataFrame(col_names, columns=['Names'], index=col_names)
    dtyp = pd.DataFrame(data.dtypes, columns=['Type'])
    misval = pd.DataFrame(data.isnull().sum(), columns=['Missing_values'])
    presval = pd.DataFrame(data.count(), columns=['Present_values'])
    unival = pd.DataFrame(columns=['Unique_values'])
    minval = pd.DataFrame(columns=['Min_value'])
    maxval = pd.DataFrame(columns=['Max_value'])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        minval.loc[col] = [data[col].min()]
        maxval.loc[col] = [data[col].max()]
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)


#%% Import the dataset and take a first look of it
data_raw = pd.read_csv(r'C:\01_iteso\analisis_pred\tareas\data\audit_risk.csv', sep=',')
report_data = dqr(data_raw)

#%% Data cleansing
data_clean = data_raw[data_raw.LOCATION_ID.str.isnumeric()]
data_clean = data_clean[data_clean.Money_Value.notna()]
data_clean.pop('Detection_Risk')
report_clean_data = dqr(data_clean)

#%% Scale the Data
data_clean_copy = data_clean.copy()
y_values = data_clean_copy.pop('Risk')
x_values = data_clean_copy.copy()

scaler = StandardScaler().fit(x_values)
x_values_sc = scaler.transform(x_values)
x_values_sc = pd.DataFrame(x_values_sc, columns=list(x_values.columns))

x_train, x_test, y_train, y_test = train_test_split(x_values_sc, y_values, test_size=0.3, random_state=11235)

#%% Logistic Regression
n_grado = 2
poly_features = PolynomialFeatures(n_grado)

xa_test = poly_features.fit_transform(x_test)
xa_train = poly_features.fit_transform(x_train)

reg_log_model = linear_model.LogisticRegression(C=1)
reg_log_model.fit(xa_train, y_train)

# View the overfiting by means the model coefficients
coef_model = reg_log_model.coef_
plt.bar(np.arange(len(coef_model[0])), coef_model[0])
plt.show()

# Model prediction and scores
y_predict_reg_log = reg_log_model.predict(xa_test)
print(f'Accuracy score = {round(accuracy_score(y_test, y_predict_reg_log), 3)}')
print(f'Precision score = {round(precision_score(y_test, y_predict_reg_log), 3)}')
print(f'Recall score = {round(recall_score(y_test, y_predict_reg_log), 3)}')

#%% Linear Regression with Variable Elimination

# Variable Elimination (Correlation Criteria)
data_reduction = x_values_sc.copy()
while True:
    temp_corr_matrix = data_reduction.corr()
    corr_matrix_aux = temp_corr_matrix.replace(1, -1)
    max_value = corr_matrix_aux.to_numpy().max()
    max_value_index = np.unravel_index(corr_matrix_aux.to_numpy().argmax(),
                                       corr_matrix_aux.to_numpy().shape)
    if max_value >= 0.75:
        max_var = 0
        if max_value_index[1] > max_value_index[0]:
            max_var = 1
        key_value = data_reduction.keys()[max_value_index[max_var]]
        data_reduction = data_reduction.drop(key_value, axis=1)
    else:
        break

x_test_red = x_test.copy()
x_train_red = x_train.copy()
selected_var = list(data_reduction.columns.values)

for var in selected_var:
    x_test_red.pop(var)
    x_train_red.pop(var)

# Model prediction and scores
lin_regr_model = LinearRegression().fit(x_train_red, y_train)
y_predict_lin_reg = lin_regr_model.predict(x_test_red)

y_predict_lin_reg[y_predict_lin_reg < 0.5] = 0
y_predict_lin_reg[y_predict_lin_reg >= 0.5] = 1

print(f'Accuracy score = {round(accuracy_score(y_test, y_predict_lin_reg), 3)}')
print(f'Precision score = {round(precision_score(y_test, y_predict_lin_reg), 3)}')
print(f'Recall score = {round(recall_score(y_test, y_predict_lin_reg), 3)}')

#%% Support Vector Classification (all dimensions)

svc_model = svm.SVC(kernel='linear', C=10)
svc_model.fit(x_train, y_train)
y_predict_svc = svc_model.predict(x_test)

# Model prediction and scores
print(f'Accuracy score = {round(accuracy_score(y_test, y_predict_svc), 3)}')
print(f'Precision score = {round(precision_score(y_test, y_predict_svc), 3)}')
print(f'Recall score = {round(recall_score(y_test, y_predict_svc), 3)}')

#%% Support Vector Classification (variable reduction)

svc_model_red = svm.SVC(kernel='linear', C=10)
svc_model_red.fit(x_train_red, y_train)
y_predict_svc_red = svc_model_red.predict(x_test_red)

# Model prediction and scores
print(f'Accuracy score = {round(accuracy_score(y_test, y_predict_svc_red), 3)}')
print(f'Precision score = {round(precision_score(y_test, y_predict_svc_red), 3)}')
print(f'Recall score = {round(recall_score(y_test, y_predict_svc_red), 3)}')
