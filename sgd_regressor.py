from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from funcs import criar_features_numpy
import os
import joblib

print("📦 Carregando dados de treino...")
data = np.load("X_y_treino.npz")
X = data["X"]
y = data["y"]
X_treino_feat = criar_features_numpy(X, y)
print("\n Treinando modelo SGD...")
# Pipeline com normalização (muito importante para SGD)
model = make_pipeline(
    StandardScaler(),
    SGDRegressor(
        max_iter=1000,       # número de iterações (passadas completas)
        tol=1e-3,            # critério de parada
        penalty='l1',        # regularização (ridge)
        alpha=0.0001,        # força da regularização
        learning_rate='adaptive',
        eta0=0.01,           # taxa inicial de aprendizado
        verbose=1,
    )
)

model.fit(X_treino_feat, y)
os.makedirs("modelos", exist_ok=True)
nome_modelo = "modelos/sgd_with_datas_festures_l1_features.joblib"
joblib.dump(model, nome_modelo)
print(f"💾 Modelo salvo em: {nome_modelo}")
print("✅ Modelo treinado com sucesso!")
