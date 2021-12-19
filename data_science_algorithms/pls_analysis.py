import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression


# %% DATA QUALITY REPORT
def dqr(data):
    # List of database variables
    cols = pd.DataFrame(list(data.columns.values), columns=['Names'], index=list(data.columns.values))
    # List of data types
    dtyp = pd.DataFrame(data.dtypes, columns=['Type'])
    # List of missing data
    misval = pd.DataFrame(data.isnull().sum(), columns=['Missing_values'])
    # List of present data
    presval = pd.DataFrame(data.count(), columns=['Present_values'])
    # List of unique values
    unival = pd.DataFrame(columns=['Unique_values'])
    # List of min values
    minval = pd.DataFrame(columns=['Min_value'])
    # List of max values
    maxval = pd.DataFrame(columns=['Max_value'])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    # Join the tables and return the result
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)


def compute_linear_regression(xdata, ydata):
    x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=11235)
    linear_regression = LinearRegression().fit(x_train, y_train)
    y_train_pred = linear_regression.predict(x_train)
    y_test_pred = linear_regression.predict(x_test)

    r_score_train = linear_regression.score(x_train, y_train)
    rsme_train = mean_squared_error(y_train, y_train_pred)
    print(r_score_train, rsme_train)

    r_score_test = linear_regression.score(x_test, y_test)
    rsme_test = mean_squared_error(y_test, y_test_pred)
    print(r_score_test, rsme_test)

    ref = np.linspace(min(y_test), max(y_test))
    plt.scatter(y_test,y_test_pred)
    plt.plot(ref, ref, 'k--')
    plt.axis('square')
    plt.xlabel('y real'), plt.ylabel('y predict')
    plt.title('Linear regression (original), RMSE=%0.4f, R^2=%0.4f' % (rsme_test, r_score_test))
    plt.grid()
    plt.show()


#%% IMPORT THE DATA SET
data_raw = pd.read_csv(r'C:\01_iteso\analisis_pred\tareas\data\airfoil_self_noise.dat', header=None, sep='\t')
col_names = ['frequency', 'angle_attack', 'chord_len', 'free_str_vel', 'suct_side', 'decibels']
data_raw.columns = col_names

print(data_raw.dtypes)
print(data_raw.describe())

#%% Obtaining the data quality report
report_nan = dqr(data_raw)

#%% Spliting the data
y = data_raw.pop('decibels')
x = data_raw.copy()

scaler = StandardScaler().fit(x)
x = scaler.transform(x)
x = pd.DataFrame(x, columns=col_names[:-1])

#%% Linear Regression
compute_linear_regression(x, y)

#%% Elimination of variables (Correlation Criteria)
data_reduction = x.copy()
while True:
    corr_matrix = data_reduction.corr()
    corr_matrix_aux = corr_matrix.replace(1, -1)
    max_value = corr_matrix_aux.to_numpy().max()
    max_value_index = np.unravel_index(corr_matrix_aux.to_numpy().argmax(), corr_matrix_aux.to_numpy().shape)

    print(corr_matrix)
    print(max_value)
    print(max_value_index)

    if max_value >= 0.75:
        max_var = 0
        if max_value_index[1] > max_value_index[0]:
            max_var = 1
        print(max_value_index[max_var])
        key_value = data_raw.keys()[max_value_index[max_var]]
        data_reduction = data_raw.drop(key_value, axis=1)
    else:
        break

#%% Linear Regression (after correlation criteria)
compute_linear_regression(data_reduction, y)

#%% Elimination of variables (PCA)
pca = PCA()
pca.fit(x)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

data_pca = pca.transform(x)
data_pca = pd.DataFrame(data_pca)
data_pca = data_pca[data_pca.columns[1:4]]
data_pca.columns = ['x1', 'x2', 'x3']

#%% Linear Regression (after pca)
compute_linear_regression(data_pca, y)

#%% Elimination of variables (PLS)
pls = PLSRegression(n_components=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=11235)
pls.fit(x_train, y_train)

data_pls = pls.transform(x)
data_pls = pd.DataFrame(data_pls,columns=['x1', 'x2', 'x3', 'x4'])
compute_linear_regression(data_pls, y)
