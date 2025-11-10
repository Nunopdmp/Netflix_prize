import os
import numpy as np
from datetime import date
import joblib  # usado para salvar o modelo
from time import time
from funcs import criar_features_numpy
import json

# ---------------------------------------------------------
# 1. Importa o modelo certo (GPU ou CPU)
# ---------------------------------------------------------
try:
    from cuml.ensemble import RandomForestClassifier as RFClassifier
    gpu_available = True
    print("✅ Usando GPU (cuML)")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as RFClassifier
    gpu_available = False
    print("⚙️  Usando CPU (scikit-learn)")


# ---------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------


# ---------------------------------------------------------
# 3. Treinamento do modelo
# ---------------------------------------------------------
def treinar_random_forest():
    print("📦 Carregando dados de treino...")
    data = np.load("X_y_treino.npz")
    X = data["X"]
    y = data["y"]
    X_treino_feat = criar_features_numpy(X, y)

    print(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")

    print("\n🌲 Treinando modelo Random Forest...")
    inicio = time()
    model = RFClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        verbose=False
    )
    model.fit(X_treino_feat, y)
    fim = time()
    print(f"✅ Treinamento concluído em {fim - inicio:.2f} segundos")

    # -----------------------------------------------------
    # 4. Salvar modelo treinado
    # -----------------------------------------------------
    os.makedirs("modelos", exist_ok=True)
    nome_modelo = "modelos/random_forest_with_datas_festures_gpu.joblib" if gpu_available else "modelos/random_forest_cpu.joblib"
    joblib.dump(model, nome_modelo)
    print(f"💾 Modelo salvo em: {nome_modelo}")


# ---------------------------------------------------------
# Execução
# ---------------------------------------------------------
if __name__ == "__main__":
    treinar_random_forest()
