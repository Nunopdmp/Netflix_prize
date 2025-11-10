import joblib
import numpy as np
from funcs import cria_X_teste, criar_features_numpy
from sklearn.metrics import mean_squared_error
    
# caminho onde você salvou
modelo = joblib.load("modelos/random_forest_gpu.joblib")
print("📦 Carregando dados de treino...")
data = np.load("X_y_treino.npz")
X = data["X"]
y = data["y"]

y = np.asarray(y, dtype=np.uint8)
print("\n🌲 Prevendo com o modelo Random Forest sem datas")
y_pred = modelo.predict(X)
print("\n✅ Avaliação concluída.")

modelo = joblib.load("modelos/random_forest_with_datas_gpu.joblib")
print("\n🌲 Prevendo com o modelo Random Forest com datas")
y_pred_with_dates = modelo.predict(X)
print("\n✅ Avaliação concluída.")

modelo = joblib.load("modelos/sgd_with_datas_festures_l2.joblib")
print("\nPrevendo com o modelo SGD com datas e features adicionais")
y_pred_gd_l2 = modelo.predict(X)
print("\n✅ Avaliação concluída.")

modelo = joblib.load("modelos/sgd_with_datas_festures_l1.joblib")
print("\nPrevendo com o modelo SGD com datas e features adicionais")
y_pred_gd_l1 = modelo.predict(X)
print("\n✅ Avaliação concluída.")

modelo = joblib.load("modelos/sgd_with_datas_festures_l1_features.joblib")
print("\nPrevendo com o modelo SGD com datas e features adicionais")
y_pred_gd_l1_features = modelo.predict(criar_features_numpy(X, y))

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mse_with_dates = mean_squared_error(y, y_pred_with_dates)
rmse_with_dates = np.sqrt(mse_with_dates)
gsd_mse_l2 = mean_squared_error(y, y_pred_gd_l2)
gsd_rmse_l2 = np.sqrt(gsd_mse_l2)
gsd_mse_l1 = mean_squared_error(y, y_pred_gd_l1)
gsd_rmse_l1 = np.sqrt(gsd_mse_l1)
gsd_mse_l1_features = mean_squared_error(y, y_pred_gd_l1_features)
gsd_rmse_l1_features = np.sqrt(gsd_mse_l1_features)


print("MSE:", mse, "RMSE:", rmse)
print("MSE com datas:", mse_with_dates, "RMSE com datas:", rmse_with_dates)
print("MSE SGD com l2:", gsd_mse_l2, "RMSE SGD com l2:", gsd_rmse_l2)
print("MSE SGD com l1:", gsd_mse_l1, "RMSE SGD com l1:", gsd_rmse_l1)
print("MSE SGD com l1 e features adicionais:", gsd_mse_l1_features, "RMSE SGD com l1 e features adicionais:", gsd_rmse_l1_features)
