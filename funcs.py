import json
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def ler_json_para_df(caminho_json: str):
    """
    Lê um arquivo JSON no formato:
        {
          "ID_filme": ["ID_cliente,avaliacao,ano_ou_data", ...],
          ...
        }
    e retorna um DataFrame com colunas:
        ID_filme (int), ID_cliente (int), avaliacao (int), ano_da_avaliacao (datetime64)
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_cliente_list = []
    id_filme_list = []
    notas_list = []
    data_aval_list = []
    for id_filme, avaliacoes in data.items():
        for registro in avaliacoes:
            partes = [p.strip() for p in str(registro).split(",")]
            if len(partes) != 3:
                continue  # pula linhas malformadas

            id_cliente_str, nota_str, data_str = partes

            try:
                id_cliente = int(id_cliente_str)
                nota = int(nota_str)  # força int
                data_aval = str(data_str)
            except ValueError:
                continue  # pula linhas com dados inválidos
            id_cliente_list.append(id_cliente)
            id_filme_list.append(int(id_filme))
            notas_list.append(nota)
            data_aval_list.append(data_aval)

    return id_cliente_list, id_filme_list, notas_list, data_aval_list

from datetime import datetime, timezone

def datas_str_para_float_fast(lista_datas):
    """
    Converte ['YYYY-MM-DD', ...] -> [timestamp_float, ...]
    Remove os elementos de lista_datas (destrutivo).
    """
    out = []
    append = out.append
    pop = lista_datas.pop  # pop do fim é O(1)

    while lista_datas:
        s = pop()  # pega último elemento
        try:
            y = int(s[0:4]); m = int(s[5:7]); d = int(s[8:10])
            ts = datetime(y, m, d, tzinfo=timezone.utc).timestamp()
        except Exception:
            ts = None  # ou continue/0.0, se preferir
            print(f'Data inválida encontrada: {s}')
        append(ts)

    out.reverse()  # reverte para manter a ordem original
    return out

def cria_X_teste():
    id_cliente_list_1, id_filme_list_1, notas_list_1, data_aval_list_1 = ler_json_para_df("dic_teste.json")
    id_cliente_list_2, id_filme_list_2, notas_list_2, data_aval_list_2 = ler_json_para_df("dic_teste_2.json")
    id_cliente_list_3, id_filme_list_3, notas_list_3, data_aval_list_3 = ler_json_para_df("dic_teste_3.json")
    id_cliente_list_4, id_filme_list_4, notas_list_4, data_aval_list_4 = ler_json_para_df("dic_teste_4.json")

    # concatena
    id_cliente_final = id_cliente_list_1 + id_cliente_list_2 + id_cliente_list_3 + id_cliente_list_4
    id_filme_final   = id_filme_list_1   + id_filme_list_2   + id_filme_list_3   + id_filme_list_4
    notas_final      = notas_list_1      + notas_list_2      + notas_list_3      + notas_list_4
    data_aval_final  = data_aval_list_1  + data_aval_list_2  + data_aval_list_3  + data_aval_list_4

    # sanity check
    if not (len(id_cliente_final) == len(id_filme_final) == len(notas_final) == len(data_aval_final)):
        raise ValueError("As listas não têm o mesmo tamanho!")

    # converte data
    data_aval_float = datas_str_para_float_fast(data_aval_final)

    # pega df de filmes (1x)
    df = df_lancamento_filmes()

    # cria dict id_filme -> ano
    mapa_ano = dict(zip(df["ID_filme"].astype(int), df["ano_de_lancamento"].astype(int)))

    # agora monta o vetor de anos com dict lookup
    anos_lanc = [mapa_ano.get(fid, 0) for fid in id_filme_final]  # 0 se não achar

    # converte para np.array
    ID_filme = np.asarray(id_filme_final, dtype=np.int32)
    ID_cliente = np.asarray(id_cliente_final, dtype=np.int32)
    dias_epoch = np.asarray(data_aval_float, dtype=np.int32)
    anos_lanc = np.asarray(anos_lanc, dtype=np.int32)

    n = len(ID_filme)

    X = np.empty((n, 4), dtype=np.int32)
    X[:, 0] = ID_filme
    X[:, 1] = ID_cliente
    X[:, 2] = dias_epoch
    X[:, 3] = anos_lanc

    return X, notas_final

def cria_X_treino():
    # lê os 4 pedaços
    id_cliente_list_1, id_filme_list_1, notas_list_1, data_aval_list_1 = ler_json_para_df("dic_treino.json")
    id_cliente_list_2, id_filme_list_2, notas_list_2, data_aval_list_2 = ler_json_para_df("dic_treino_2.json")
    id_cliente_list_3, id_filme_list_3, notas_list_3, data_aval_list_3 = ler_json_para_df("dic_treino_3.json")
    id_cliente_list_4, id_filme_list_4, notas_list_4, data_aval_list_4 = ler_json_para_df("dic_treino_4.json")

    # concatena
    id_cliente_final = id_cliente_list_1 + id_cliente_list_2 + id_cliente_list_3 + id_cliente_list_4
    id_filme_final   = id_filme_list_1   + id_filme_list_2   + id_filme_list_3   + id_filme_list_4
    notas_final      = notas_list_1      + notas_list_2      + notas_list_3      + notas_list_4
    data_aval_final  = data_aval_list_1  + data_aval_list_2  + data_aval_list_3  + data_aval_list_4

    # sanity check
    if not (len(id_cliente_final) == len(id_filme_final) == len(notas_final) == len(data_aval_final)):
        raise ValueError("As listas não têm o mesmo tamanho!")

    # converte data
    data_aval_float = datas_str_para_float_fast(data_aval_final)

    # pega df de filmes (1x)
    df = df_lancamento_filmes()

    # cria dict id_filme -> ano
    mapa_ano = dict(zip(df["ID_filme"].astype(int), df["ano_de_lancamento"].astype(int)))

    # agora monta o vetor de anos com dict lookup
    anos_lanc = [mapa_ano.get(fid, 0) for fid in id_filme_final]  # 0 se não achar

    # converte para np.array
    ID_filme = np.asarray(id_filme_final, dtype=np.int32)
    ID_cliente = np.asarray(id_cliente_final, dtype=np.int32)
    dias_epoch = np.asarray(data_aval_float, dtype=np.int32)
    anos_lanc = np.asarray(anos_lanc, dtype=np.int32)

    n = len(ID_filme)

    X = np.empty((n, 4), dtype=np.int32)
    X[:, 0] = ID_filme
    X[:, 1] = ID_cliente
    X[:, 2] = dias_epoch
    X[:, 3] = anos_lanc

    return X, notas_final

    
def df_lancamento_filmes():
    df1 = pd.read_csv(
        'kaggle/input/movie_titles.csv',
        delimiter=',',
        encoding='latin1'
    )
    df1.dataframeName = 'movie_titles.csv'

    # Renomear colunas conforme estrutura original do dataset Netflix
    df1 = df1.rename(columns={
        '0': 'ID_filme',
        '1': 'ano_de_lancamento',
        '2': 'nome'
    })

    # Converter para numérico (qualquer coisa inválida vira NaN)
    df1["ID_filme"] = pd.to_numeric(df1["ID_filme"], errors="coerce")
    df1["ano_de_lancamento"] = pd.to_numeric(df1["ano_de_lancamento"], errors="coerce")

    # Remover linhas sem ID
    df1 = df1.dropna(subset=["ID_filme"])

    # Calcular média dos anos válidos
    media_anos = int(df1["ano_de_lancamento"].mean(skipna=True))

    # Preencher anos ausentes com a média
    df1["ano_de_lancamento"] = df1["ano_de_lancamento"].fillna(media_anos).astype(int)

    # Converter IDs para int
    df1["ID_filme"] = df1["ID_filme"].astype(int)

    return df1

import numpy as np

def criar_features_numpy(X, y):
    """
    X: matriz (n, 4)
        X[:,0] = id_filme
        X[:,1] = id_cliente
        X[:,2] = dias_epoch
        X[:,3] = ano_lanc (pode ter 0)
    y: vetor de notas (n,)
    """
    # garante tipos
    id_filme   = X[:, 0].astype(np.int64)
    id_cliente = X[:, 1].astype(np.int64)
    dias_epoch = X[:, 2].astype(np.float32)
    ano_lanc   = X[:, 3].astype(np.float32)

    # 1) tratar ano faltando (0) -> média dos que têm ano
    mask_zero = (ano_lanc == 0)
    if np.any(mask_zero):
        media_ano = ano_lanc[~mask_zero].mean()
        ano_lanc[mask_zero] = media_ano

    # 2) estatísticas por filme
    max_filme = id_filme.max()
    soma_por_filme = np.bincount(id_filme, weights=y, minlength=max_filme+1)
    cont_por_filme = np.bincount(id_filme, minlength=max_filme+1)
    media_por_filme = soma_por_filme / np.maximum(cont_por_filme, 1)

    # traz para o nível da amostra
    feat_media_filme = media_por_filme[id_filme]
    feat_cont_filme  = cont_por_filme[id_filme]

    # 3) estatísticas por cliente
    max_cliente = id_cliente.max()
    soma_por_cliente = np.bincount(id_cliente, weights=y, minlength=max_cliente+1)
    cont_por_cliente = np.bincount(id_cliente, minlength=max_cliente+1)
    media_por_cliente = soma_por_cliente / np.maximum(cont_por_cliente, 1)

    feat_media_cliente = media_por_cliente[id_cliente]
    feat_cont_cliente  = cont_por_cliente[id_cliente]

    # 4) interação user x filme
    diff_user_movie = feat_media_cliente - feat_media_filme

    # 5) tempo desde lançamento (em dias aproximados)
    tempo_desde_lanc = dias_epoch - (ano_lanc - 1970.0) * 365.0

    # 6) opcional: comprimir contagens (senão vira número muito grande)
    feat_cont_filme_log   = np.log1p(feat_cont_filme)
    feat_cont_cliente_log = np.log1p(feat_cont_cliente)

    # 7) montar matriz final
    # você pode escolher quais manter; vou manter as originais + derivadas
    X_new = np.column_stack([
        id_filme,
        id_cliente,
        dias_epoch,
        ano_lanc,
        feat_media_filme,
        feat_media_cliente,
        feat_cont_filme_log,
        feat_cont_cliente_log,
        diff_user_movie,
        tempo_desde_lanc,
    ])

    return X_new
