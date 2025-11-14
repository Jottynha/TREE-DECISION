import numpy as np
from preprocess import preprocess_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#optuna svm scikit_learn

input_file = 'data/raw/Watera.csv'

X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X_original = preprocess_data(input_file)

# Padronizar os dados (muito importante para PCA e SVM)
X_train = (X_train_scaled)
X_test = (X_test_scaled)

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=2)  # Reduzindo para 2 componentes principais
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Treinar o modelo SVM
svm = SVC(kernel='linear')  # Você pode escolher outros kernels como 'rbf'
svm.fit(X_train_pca, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm.predict(X_test_pca)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

import pickle
# Save the model to a file
with open('svm.model', 'wb') as file:
  pickle.dump(svm, file)

# Load the model from the file
with open('svm.model', 'rb') as file:
  svm = pickle.load(file)
  
  
# treinar o model com cross-validation
from sklearn.model_selection import cross_val_score
from preprocess import get_kfold_splits
svm_cv = SVC(kernel='linear')
cv = get_kfold_splits(5,42)
scores = cross_val_score(svm_cv, X_train_pca, y_train, cv=cv)
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores):.2f}')
# fazer as predicoes com o modelo treinado com cross-validation
y_pred_cv = svm_cv.fit(X_train_pca, y_train).predict(X_test_pca)
accuracy_cv = accuracy_score(y_test, y_pred_cv)
print(f'Acurácia com cross-validation: {accuracy_cv:.2f}')