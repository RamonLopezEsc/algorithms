import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_curve, auc


pd.set_option('display.max_rows', None)


def pre_process_data():
    data = pd.read_csv(r'C:\01_iteso\analisis_pred\proyecto\xlsx\pr'
                       r'oj_dat_reduced.csv', sep=',')
    pop_col_dict = {'db_id': None, 'spotify_id': None, 'principal_artist': None,
                    'album': None, 'track_name': None}
    for keycol in pop_col_dict:
        pop_col_dict[keycol] = data.pop(keycol)

    data_process = data.copy()
    data_process.pop('acousticness')
    data_process.pop('instrumentalness')

    data_process['tempo'] = preprocessing.minmax_scale(data_process['tempo'])
    data_process['loudness'] = preprocessing.minmax_scale(data_process['loudness'])
    data_process['popularity'] = preprocessing.minmax_scale(data_process['popularity'])

    data_process['loudness'] = data_process['loudness'] + 1
    data_process['loudness'] = preprocessing.power_transform(data_process['loudness'].
                                                             to_numpy().reshape(-1, 1),
                                                             method='box-cox')

    valence_values = data_process.pop('user')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(valence_values)
    valence_values_labels = label_encoder.transform(valence_values)

    return data, data_process, valence_values_labels


data_raw, x_data, y_data = pre_process_data()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,
                                                    random_state=112358, stratify=y_data)


# Logistic Regression
print('---------------------------------------------------------------------------'
      '-----------------------------------')
print('Logistic Regression')

# C -> Regularizacion 1/c (c = 1, regulariza/sobre entrena)
log_reg_model = linear_model.LogisticRegression(C=1, max_iter=10000)

# Recordar -> dim <= num datos
n_grado = 3
poly_features = preprocessing.PolynomialFeatures(n_grado)

xa_test = poly_features.fit_transform(x_test)
xa_train = poly_features.fit_transform(x_train)

log_reg_model.fit(xa_train, y_train)
y_predict_log_reg = log_reg_model.predict(xa_test)

# Modelo sobre entrenado -> Se espera que los parametros originales sean los mas "pesados"
coef_model = log_reg_model.coef_
plt.bar(np.arange(len(coef_model[0])), coef_model[0])
plt.show()

print('Traing')
y_predict_log_train = log_reg_model.predict(xa_train)
print(f'Accuracy score = {accuracy_score(y_train, y_predict_log_train)}')
print(f'Precision score = {precision_score(y_train, y_predict_log_train)}')
print(f'Recall score = {recall_score(y_train, y_predict_log_train)}')

print('\nTest')
print(f'Accuracy score = {accuracy_score(y_test, y_predict_log_reg)}')
print(f'Precision score = {precision_score(y_test, y_predict_log_reg)}')
print(f'Recall score = {recall_score(y_test, y_predict_log_reg)}')

# Support Vector Machines
print('---------------------------------------------------------------------------'
      '-----------------------------------')
print('Support Vector Machines')

svc_model = svm.SVC(kernel='linear', C=1, probability=True)
svc_model.fit(x_train, y_train)
y_predict_svc = svc_model.predict(x_test)

print('Traing')
y_predict_svm_train = svc_model.predict(x_train)
print(f'Accuracy score = {accuracy_score(y_train, y_predict_svm_train)}')
print(f'Precision score = {precision_score(y_train, y_predict_svm_train)}')
print(f'Recall score = {recall_score(y_train, y_predict_svm_train)}')

print('\nTest')
print(f'Accuracy score = {accuracy_score(y_test, y_predict_svc)}')
print(f'Precision score = {precision_score(y_test, y_predict_svc)}')
print(f'Recall score = {recall_score(y_test, y_predict_svc)}')

# MLP
print('---------------------------------------------------------------------------'
      '-----------------------------------')
print('Neural Network')

dummy_test = np_utils.to_categorical(y_test)[:, :]
dummy_train = np_utils.to_categorical(y_train)[:, :]

model = Sequential()
model.add(Dense(x_train.shape[0] * x_train.shape[0], activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))

# Optimizer configuration
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Neural network training
# es_callback = EarlyStopping(monitor='val_loss', patience=100)
model_history = model.fit(x_train, dummy_train, epochs=250, batch_size=1, verbose=1,
                          validation_data=(x_test, dummy_test))

fig_acc, ax_acc = plt.subplots(1, 1, figsize=(10, 6))
ax_acc.plot(model_history.history['accuracy'], 'r', label='train')
ax_acc.plot(model_history.history['val_accuracy'], 'b', label='val')
ax_acc.set_xlabel('epoch')
ax_acc.set_ylabel('accuracy')
ax_acc.legend()
plt.show()

fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 6))
ax_loss.plot(model_history.history['loss'], 'r', label='train')
ax_loss.plot(model_history.history['val_loss'], 'b', label='val')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
plt.show()

print('---------------------------------------------------------------------------'
      '-----------------------------------')
print('Results / Predictions')

y_prob_train = model.predict(x_train)
y_pred_train = np.argmax(y_prob_train, axis=1)

y_prob_test = model.predict(x_test)
y_pred_test = np.argmax(y_prob_test, axis=1)

score = model.evaluate(x_test, dummy_test, verbose=1)
print(score)

accu_train = accuracy_score(y_train, y_pred_train)
prec_train = precision_score(y_train, y_pred_train)
reca_train = recall_score(y_train, y_pred_train)

accu_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test)
reca_test = recall_score(y_test, y_pred_test)

print('Train...')
print(f'Accuracy: {accu_train}')
print(f'Precision: {prec_train}')
print(f'Recall: {reca_train}')

print('\nTest...')
print(f'Accuracy: {accu_test}')
print(f'Precision: {prec_test}')
print(f'Recall: {reca_test}')

print('---------------------------------------------------------------------------'
      '-----------------------------------')
print('Saving the model...')
model_json = model.to_json()

with open("model_mlp_reduced.json", "w") as json_file:
    json_file.write(model_json)

model.save("model_mlp_reduced.h5")
print("Model saved")

print('---------------------------------------------------------------------------'
      '-----------------------------------')

# ROC curve and AUC

# Logistic Regression
fpr_train_log, tpr_train_log, thresholds_train_log = roc_curve(y_train,
                                                               y_predict_log_train)
fpr_test_log, tpr_test_log, thresholds_test_log = roc_curve(y_test, y_predict_log_reg)
auc_train_log, auc_test_log = auc(fpr_train_log, tpr_train_log), auc(fpr_test_log,
                                                                     tpr_test_log)

# SVM
fpr_train_svm, tpr_train_svm, thresholds_train_svm = roc_curve(y_train,
                                                               y_predict_svm_train)
fpr_test_svm, tpr_test_svm, thresholds_test_svm = roc_curve(y_test, y_predict_svc)
auc_train_svm, auc_test_svm = auc(fpr_train_svm, tpr_train_svm), auc(fpr_test_svm,
                                                                     tpr_test_svm)

# Neural Network
fpr_train_nn, tpr_train_nn, thresholds_train_nn = roc_curve(y_train, y_pred_train)
fpr_test_nn, tpr_test_nn, thresholds_test_nn = roc_curve(y_test, y_pred_test)
auc_train_nn, auc_test_nn = auc(fpr_train_nn, tpr_train_nn), auc(fpr_test_nn,
                                                                 tpr_test_nn)

# Show the model comparation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr_train_nn, tpr_train_nn, label='Neural Network, (AUC= %0.3f)' %
                                           auc_train_nn)
plt.plot(fpr_train_log, tpr_train_log, label='Logistic Regression, (AUC= %0.3f)' %
                                             auc_train_log)
plt.plot(fpr_train_svm, tpr_train_svm, label='SVC (AUC= %0.3f)' % auc_train_svm)
plt.legend()
plt.title('ROC curve train')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.subplot(1, 2, 2)
plt.plot(fpr_test_nn, tpr_test_nn, label='Neural Network, (AUC= %0.3f)' %
                                         auc_test_nn)
plt.plot(fpr_test_log, tpr_test_log, label='Logistic Regression, (AUC= %0.3f)' %
                                           auc_test_log)
plt.plot(fpr_test_svm, tpr_test_svm, label='SVC (AUC= %0.3f)' % auc_test_svm)
plt.legend()
plt.title('ROC curve test')
plt.xlabel('1-specificity')
plt.ylabel('sensitivity')
plt.show()
