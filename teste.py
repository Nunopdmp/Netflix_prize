import joblib
import numpy as np
from funcs import criar_features_numpy
from sklearn.metrics import mean_squared_error

print("📦 Carregando dados de treino...")
data = np.load("X_y_treino.npz")
X = data["X"]
y = data["y"]
X_treino_feat = criar_features_numpy(X, y)
y = np.asarray(y, dtype=np.uint8)


modelo = joblib.load("modelos/sgd_with_datas_festures_l1_features.joblib")
print("\nPrevendo com o modelo SGD com datas e features adicionais")
y_pred_gd = modelo.predict(X_treino_feat)
y_pred_gd = np.round(y_pred_gd).astype(np.uint8)
print("\n✅ Avaliação concluída.")
print("y_pred_gd:", y_pred_gd[:5])
print("y:", y[:5])
mse = mean_squared_error(y, y_pred_gd)
rmse = np.sqrt(mse)
print("MSE SGD com datas e features adicionais:", mse, "RMSE SGD com datas e features adicionais:", rmse)