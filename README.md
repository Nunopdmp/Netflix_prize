# 🧠 Netflix Prize — Recriando o Desafio com ML Moderno

Este projeto foi inspirado no **Netflix Prize**, a competição lançada em **2006**, onde o objetivo era **melhorar o sistema de recomendação da Netflix** atingindo um **RMSE ≤ 0.95** — um marco que desafiou pesquisadores e engenheiros de todo o mundo.

Aqui, busquei **reproduzir esse feito** com técnicas modernas, explorando desde abordagens clássicas até frameworks otimizados por GPU, como o **cuML**, dentro de um ambiente controlado e totalmente configurado via **WSL + Python + RAPIDS**.

---

## 🎯 Objetivo

Recriar um modelo de **Machine Learning** capaz de prever as notas de usuários para filmes com **RMSE ≤ 0.95**, a mesma métrica usada no desafio original da Netflix.

---

## ⚙️ Pipeline de Desenvolvimento

Durante o processo, diversas abordagens foram testadas, desde os modelos mais simples até combinações otimizadas com feature engineering e GPU acceleration.

### 🔹 1. Ingestão e Pré-processamento
- Leitura de dezenas de milhões de registros JSON contendo:
  - `ID_filme`
  - `ID_cliente`
  - `nota`
  - `data_avaliacao`
- Criação de features adicionais:
  - **dias_epoch** — número de dias desde o início do dataset
  - **anos_lanc** — ano de lançamento do filme
  - **soma_por_filme**, **media_por_cliente** — agregações úteis para capturar padrões de comportamento
- Tratamento de valores ausentes com médias ponderadas

### 🔹 2. Modelos Testados
Durante o desenvolvimento, foram avaliados vários modelos de regressão e classificação, incluindo:

| Modelo | Observações |
|---------|--------------|
| `RandomForestRegressor` (scikit-learn) | Forte desempenho, mas alto custo computacional |
| `KNN` | Boa precisão local, mas inviável para 100M de registros |
| `SVM` (scikit + cuML) | Testado com kernel linear e RBF, sem ganho relevante |
| `SGDRegressor` | Excelente escalabilidade — base do modelo final |
| `cuML.RandomForest` | Testes acelerados em GPU, com grande redução de tempo de treinamento |

---

## 🧩 Feature Engineering

As features foram fundamentais para reduzir o erro do modelo:

```python
X[:, 0] = ID_filme
X[:, 1] = ID_cliente
X[:, 2] = dias_epoch
X[:, 3] = anos_lanc
````

Outras derivadas incluíram estatísticas agregadas, normalização e encoding leve para evitar explosão de cardinalidade.

---

## ⚡ Ambiente e Aceleração GPU

Para lidar com o volume de dados (~100 milhões de registros), foi utilizado:

* **WSL2 + Ubuntu**
* **RAPIDS + cuML**
* **GPU RTX 3060 (Laptop)**

  * Permitiu treinar modelos massivos em horas, não dias.

Apesar disso, alguns testes ainda exigiram otimizações manuais para evitar “out of memory” durante o fit.

---

## 🧮 Resultado Final

Após diversos experimentos, **o modelo baseado em `SGDRegressor` com features otimizadas** foi o suficiente para alcançar o **objetivo histórico**:

> 🎯 **RMSE = 0.95**

Esse resultado demonstra que, com **feature engineering bem estruturado**, é possível igualar o desempenho do time vencedor da Netflix — mesmo usando hardware de consumidor e frameworks modernos.

---

## 📈 Lições Aprendidas

* **Feature engineering > modelo**: pequenas features de comportamento tiveram mais impacto que trocas de algoritmo.
* **cuML e GPU** são essenciais para escalar experimentos massivos.
* **Simplicidade** vence: o modelo final (SGD + features) foi mais eficiente e estável que soluções mais complexas.
* **Gerenciamento de memória** é crítico para datasets dessa magnitude.

---

## 🧩 Próximos Passos

* Implementar validação cruzada distribuída (cuDF + Dask)
* Testar versões híbridas (SGD + embeddings de usuários/filmes)
* Comparar com arquiteturas neurais leves (MLP simples em PyTorch)

---

## 🏁 Conclusão

Recriar o **Netflix Prize** não foi apenas um exercício técnico, mas uma jornada de engenharia de dados, otimização e aprendizado contínuo.
Alcançar **RMSE = 0.95** com técnicas modernas mostra que o legado daquele desafio ainda inspira novas soluções — agora com ferramentas muito mais poderosas.

---

**Autor:** [Nuno Prado de Medeiros Paulos](https://github.com/nunopaulos)
📅 Projeto iniciado em 2025
🎓 Inspirado no Netflix Prize (2006)

```

