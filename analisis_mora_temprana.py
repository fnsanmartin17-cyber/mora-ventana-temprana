# -*- coding: utf-8 -*-
"""
===============================================================================
 PREDICCION TEMPRANA DE MORA EN CREDITOS AUTOMOTRICES
 Felipe San Martin - Magister en Data Science, Universidad San Sebastian
===============================================================================

 LA PREGUNTA
 -----------
 Cuando una financiera automotriz otorga un credito, lo normal es esperar a que
 el credito madure para saber si el cliente va a pagar. Yo quise medir otra
 cosa: cuanto se puede saber MUY temprano, mirando solo las primeras k cuotas.

 Probe con k = 1, 2, 3, 5 y 10 cuotas. Ojo con esto porque es la confusion mas
 comun: k se cuenta en CUOTAS, no en dias. "k = 5" significa "las primeras cinco
 cuotas del credito", hayan tomado cinco meses o quince.

 LO QUE HACE ESTE ARCHIVO
 ------------------------
 Todo. Carga los datos, arma las variables, entrena los seis modelos, calcula el
 clasificador trivial de referencia y escribe las tablas de los cuatro objetivos
 /.

     pip install pandas numpy scikit-learn xgboost matplotlib
     python analisis_mora_temprana.py


 LOS DATOS
 ---------
 datos_sinteticos.csv es una base SINTETICA. No hay ninguna persona real ahi.
 La base original del estudio tiene el RUT de 2.154 clientes e informacion
 comercial de la institucion que la aporto, y no se puede publicar. La sintetica
 reproduce el esquema, las distribuciones y -esto es lo importante- las cuatro
 mecanicas del dominio que explico mas abajo, para que este codigo se pueda
 correr de verdad. Los numeros que salen son del mismo orden que los publicados,
 pero NO son identicos, porque los creditos no son los mismos.
===============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
from xgboost import XGBClassifier


# =============================================================================
#  PASO 0 - Configuracion
# =============================================================================
# Fijo la semilla en todos lados. Sin esto los resultados cambian entre
# corridas y no hay forma de comparar nada.
SEMILLA = 42
np.random.seed(SEMILLA)

VENTANAS = [1, 2, 3, 5, 10]          # las k que voy a probar, en cuotas
GRUPOS_PLAZO = [12, 24, 36, 48]      # los cuatro segmentos de plazo de la cartera

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("DATOS_CSV", os.path.join(AQUI, "datos_sinteticos.csv"))
SALIDA = os.path.join(AQUI, "resultados")

METRICAS = ["Accuracy", "F1", "Recall", "Precision", "ROC_AUC", "PR_AUC"]


def titulo(txt):
    print("\n" + "=" * 78)
    print(" " + txt)
    print("=" * 78)


# =============================================================================
#  PASO 1 - Cargar los datos y entender que tengo entre manos
# =============================================================================
# El archivo viene a nivel CUOTA: una fila por cada cuota de cada credito. Un
# credito a 36 meses ocupa 36 filas. Lo que quiero predecir, en cambio, es algo
# a nivel CREDITO ("termino en mora, si o no"), asi que buena parte del trabajo
# es pasar de una granularidad a la otra sin hacer trampa.
#
# Hay cuatro cosas de estos datos que me costaron entender y que, si no se
# respetan, dan resultados equivocados. Las dejo escritas porque son el corazon
# del analisis:
#
#   1. EL PAGO ES UN PREFIJO CONTIGUO. Un credito paga las cuotas 1..L y se
#      detiene. No hay cuotas pagadas salteadas. Entonces "estar en mora"
#      equivale a que L sea menor que el total de cuotas.
#
#   2. UNA CUOTA IMPAGA NO TIENE ATRASO. Trae dias_atraso = 0 y d_pago =
#      2000-01-01, que es un centinela, no una fecha real. Si uno lee
#      dias_atraso sin filtrar por pagada == 1, los peores creditos aparecen
#      con atraso cero y todo se invierte.
#
#   3. EL PISO DE -60 DIAS ES UN PREPAGO, NO UN COMPORTAMIENTO. Cuando alguien
#      liquida el credito anticipadamente, todas las cuotas que quedaban se
#      saldan en UNA sola fecha. Como esa fecha queda congelada mientras los
#      vencimientos siguen corriendo mes a mes, el "atraso" de esas cuotas se
#      vuelve cada vez mas negativo, y el sistema lo corta en -60.
#      Consecuencia practica: en la cuota 1 el -60 no aparece en NINGUNA de las
#      dos clases, y recien emerge desde la cuota 3. Por eso k = 1 rinde como
#      el clasificador trivial y k = 5 se despega.
#
#   4. LA BASE ES UNA FOTO EN EL TIEMPO. Ver PASO 2.
def cargar():
    titulo("PASO 1 - Cargando los datos")
    if not os.path.exists(CSV):
        sys.exit("No encuentro %s" % CSV)
    df = pd.read_csv(CSV, parse_dates=["d_vencimiento", "d_pago"])
    df = df.sort_values(["c_simulacion", "c_cuota"]).reset_index(drop=True)

    n_cred = df["c_simulacion"].nunique()
    mora = df.groupby("c_simulacion")["mora"].first()
    print("  %s filas (una por cuota)" % f"{len(df):,}")
    print("  %s creditos de %s clientes distintos"
          % (f"{n_cred:,}", f"{df['a_rutcliente'].nunique():,}"))
    # Ojo con este porcentaje: la mora es la clase MAYORITARIA, no la rara. El
    # desbalance esta invertido respecto de lo habitual en riesgo de credito, y
    # eso cambia como hay que leer todas las metricas de aca en adelante.
    print("  %.1f%% terminaron en mora" % (mora.mean() * 100))
    return df


# =============================================================================
#  PASO 2 - El corte de observacion: por que la poblacion se encoge
# =============================================================================
# Esta es la cuarta mecanica, y la que mas condiciona el estudio.
#
# La base es una foto sacada en un momento dado. En ese momento, de cada credito
# solo habian vencido algunas cuotas; las demas todavia estaban en el futuro. Un
# credito otorgado hace tres meses simplemente NO TIENE una quinta cuota que yo
# pueda mirar.
#
# Entonces reconstruyo la fecha de la foto como el ultimo pago que aparece en
# toda la base, y defino que una cuota esta "observada" si ya se pago, o si
# vencio antes de esa fecha. Para cada ventana k me quedo solo con los creditos
# que tengan al menos k cuotas observadas.
#
# El efecto es fuerte: de 2.188 creditos en k = 1 quedan unos 925 en k = 10, y
# la tasa de mora cae del 81,7% al 56,8%. Y no se caen creditos al azar: se caen
# los que incumplieron temprano, que son justamente los peores. Por eso comparar
# el F1 de k = 1 contra el de k = 10 mezcla dos efectos distintos -la ventana
# mas larga y una poblacion distinta- y por eso existe el Objetivo 4.
def marcar_observadas(df):
    titulo("PASO 2 - Corte de observacion (la foto en el tiempo)")
    corte = df.loc[df["pagada"] == 1, "d_pago"].max()
    df["observada"] = ((df["pagada"] == 1) |
                       ((df["pagada"] == 0) & (df["d_vencimiento"] <= corte)))
    obs_max = df[df["observada"]].groupby("c_simulacion")["c_cuota"].max()
    print("  Fecha de la foto (ultimo pago registrado): %s" % corte.date())
    print("  Cuotas observadas por credito: mediana %d, maximo %d"
          % (obs_max.median(), obs_max.max()))
    print()
    print("  Creditos que sobreviven a cada ventana:")
    mora = df.groupby("c_simulacion")["mora"].first()
    for k in VENTANAS:
        elig = obs_max[obs_max >= k].index
        print("    k = %-2d  %4d creditos  %.1f%% mora"
              % (k, len(elig), mora.reindex(elig).mean() * 100))
    return obs_max


# =============================================================================
#  PASO 3 - Variables exogenas (lo que se sabe al firmar) y el target
# =============================================================================
# Las exogenas son las que ya se conocen el dia que se otorga el credito: precio
# del vehiculo, pie, plazo, tasa, marca, producto, seguro. Les sumo dos que
# calcule yo: el LTV (cuanto se financia sobre el precio) y el valor aproximado
# de la cuota.
#
# Descarte tres columnas a proposito: a_modelo tiene 617 niveles distintos y solo
# agrega ruido, a_fechaalta venia completamente vacia, y a_tipocliente es la
# misma palabra en todas las filas.
#
# Y hay otras que NO uso aunque vengan en el archivo, por una razon de fondo:
# cuotas_pagadas, cuotas_vencidas_np, cuotas_futuras_np, cuotas_pagadas_con_atraso,
# max_dias_atraso, tiene_atraso y estado_pago_credito resumen TODO el historial del
# credito, incluido lo que ocurre DESPUES de la ventana. Meterlas seria filtrar la
# respuesta: estado_pago_credito es literalmente la variable objetivo, y
# cuotas_pagadas la determina (si igualan a total_cuotas, el credito no esta en mora).
EXO_NUM = ["n_pie", "pct_pie", "n_plazo", "n_tasamensual",
           "n_precio", "n_totalfinanciar", "ltv", "valor_cuota_aprox"]
EXO_CAT = ["a_marca", "a_nombreproducto", "a_tipocredito", "a_seguro"]


def exogenas_y_target(df):
    titulo("PASO 3 - Variables de originacion y variable objetivo")
    base = ["n_pie", "pct_pie", "n_plazo", "n_tasamensual",
            "n_precio", "n_totalfinanciar"]
    agg = {c: "first" for c in base + EXO_CAT}
    agg["total_cuotas"] = "first"
    agg["a_rutcliente"] = "first"
    exo = df.groupby("c_simulacion").agg(agg)
    exo["ltv"] = (exo["n_totalfinanciar"] / exo["n_precio"]) \
        .replace([np.inf, -np.inf], np.nan)
    exo["valor_cuota_aprox"] = exo["n_totalfinanciar"] / exo["n_plazo"]
    exo["grupo_plazo"] = np.where(exo["total_cuotas"].isin(GRUPOS_PLAZO),
                                  exo["total_cuotas"], 0).astype(int)

    target = df.groupby("c_simulacion").agg(mora=("mora", "first"))
    print("  %d variables exogenas a nivel credito" % (len(EXO_NUM) + len(EXO_CAT)))
    print("  Clase positiva = mora (1). Reparto: %d en mora, %d pagados"
          % (int(target["mora"].sum()), int((target["mora"] == 0).sum())))
    return exo, target


# =============================================================================
#  PASO 4 - Las variables de comportamiento, sin espiar el futuro
# =============================================================================
# 
#
# El problema: si miro las primeras 5 cuotas de un credito, no puedo usar
# informacion que en ese momento todavia no existia. Un credito que se puso al
# dia en la cuota 20 no puede aparecer como "pago la cuota 3" si ese pago
# ocurrio dos años despues.
#
# La solucion: para cada credito fijo un punto de observacion propio, que es el
# vencimiento de la ultima cuota de su ventana. Y desde ahi:
#
#   - una cuota cuenta como pagada solo si el pago ocurrio ANTES de ese punto.
#     Si pago despues, en ese momento yo no lo sabria, asi que la cuento impaga.
#   - si al observar la cuota sigue impaga, su atraso no es cero: es cuantos
#     dias lleva vencida a esa fecha.
#
# Con eso armo once variables de comportamiento, todas con prefijo w_.
def racha_maxima(seq):
    """La racha mas larga de unos seguidos. Sirve para medir si los atrasos
    vienen encadenados (mala señal) o sueltos."""
    mejor = actual = 0
    for v in seq:
        actual = actual + 1 if v else 0
        mejor = max(mejor, actual)
    return mejor


def ventana_observada(df, k=None, penultima=False):
    obs = df[df["observada"]].copy()
    if penultima:
        ultima = obs.groupby("c_simulacion")["c_cuota"].transform("max")
        obs = obs[obs["c_cuota"] <= ultima - 1]
    elif k is not None:
        obs = obs[obs["c_cuota"] <= k]
    obs = obs.sort_values(["c_simulacion", "c_cuota"])

    # el punto de observacion de cada credito
    corte = obs.groupby("c_simulacion")["d_vencimiento"].transform("max")
    obs["corte_ventana"] = corte
    obs["pagada_obs"] = ((obs["pagada"] == 1) & (obs["d_pago"] <= corte)).astype(int)
    obs["atraso_efectivo"] = np.where(
        obs["pagada_obs"] == 1,
        obs["dias_atraso"].astype(float),                    # atraso real al pagar
        (corte - obs["d_vencimiento"]).dt.days.astype(float))  # dias vencida al observar
    obs["atraso_pos"] = obs["atraso_efectivo"].clip(lower=0)
    obs["es_atraso"] = (obs["atraso_efectivo"] > 0).astype(int)
    return obs


def features_ventana(df, k=None, penultima=False):
    obs = ventana_observada(df, k=k, penultima=penultima)
    g = obs.groupby("c_simulacion")
    f = pd.DataFrame(index=g.size().index)

    f["w_n_cuotas"] = g.size()                       # cuantas cuotas mire
    f["w_n_pagadas"] = g["pagada_obs"].sum()         # cuantas estaban pagadas
    f["w_n_impagas"] = f["w_n_cuotas"] - f["w_n_pagadas"]
    f["w_tasa_pago"] = f["w_n_pagadas"] / f["w_n_cuotas"]
    f["w_atraso_max"] = g["atraso_pos"].max()
    f["w_atraso_medio"] = g["atraso_pos"].mean()
    f["w_atraso_std"] = g["atraso_pos"].std()
    f["w_n_atrasos"] = g["es_atraso"].sum()
    f["w_tasa_atraso"] = f["w_n_atrasos"] / f["w_n_cuotas"]
    f["w_racha_atraso"] = g["es_atraso"].apply(lambda s: racha_maxima(s.values))

    # el atraso de la primera cuota, por si solo, ya dice bastante
    c1 = obs[obs["c_cuota"] == 1].set_index("c_simulacion")["atraso_pos"]
    f["w_atraso_cuota1"] = c1.reindex(f.index)

    # cuanto se adelanta en promedio quien paga antes de tiempo
    pag = obs[obs["pagada_obs"] == 1].copy()
    pag["anticipo"] = pag["dias_atraso"].clip(upper=0)
    f["w_anticipo_medio"] = pag.groupby("c_simulacion")["anticipo"].mean().reindex(f.index)

    return f.fillna(0.0)


def armar_datasets(df, exo, target, obs_max):
    titulo("PASO 4 - Armando las variables de comportamiento")
    DS = {}
    for k in VENTANAS:
        elig = obs_max[obs_max >= k].index
        fw = features_ventana(df, k=k)
        idx = exo.index.intersection(elig)
        ds = exo.join(fw, how="inner").join(target, how="inner").loc[idx].copy()
        DS[k] = ds
        print("  k = %-2d  %4d creditos  %.1f%% mora  %d variables"
              % (k, len(ds), ds["mora"].mean() * 100, ds.shape[1] - 1))

    # El dataset del Objetivo 1: todo el historial hasta la penultima cuota.
    # Es el "techo" de lo que se podria saber, no una prediccion util: usar la
    # penultima cuota para predecir el desenlace es casi mirar la respuesta.
    ds_techo = (exo.join(features_ventana(df, penultima=True), how="inner")
                   .join(target, how="inner"))
    print("  techo   %4d creditos  %.1f%% mora  (hasta la penultima cuota)"
          % (len(ds_techo), ds_techo["mora"].mean() * 100))
    return DS, ds_techo


# =============================================================================
#  PASO 5 - El clasificador trivial, que es la vara real
# =============================================================================
# Esta parte es la que mas me cambio la lectura del trabajo.
#
# Si la mora es el 81,7% de los casos, un clasificador que responde "mora"
# siempre, sin mirar un solo dato, saca:
#
#       Accuracy  = p
#       Recall    = 1        (acierta todos los morosos, obvio: dice que todos lo son)
#       Precision = p
#       F1        = 2p / (1 + p)
#
# Con p = 0,817 eso da F1 = 0,899. Un modelo con F1 0,90 no esta prediciendo
# nada: esta empatando con no hacer nada. Por eso todas las tablas de este
# trabajo reportan el trivial al lado del modelo.
#
# Un detalle que vale la pena: esa formula vale SOLO mientras la clase
# mayoritaria sea la mora. En el segmento de 12 cuotas con ventana k = 5 la mora
# baja al 49,3%, la mayoritaria pasa a ser "pagado", el trivial deja de predecir
# mora y su F1 sobre la clase mora se desploma a 0,221. Ahi es donde los modelos
# mas claramente le ganan, aunque su F1 absoluto sea el mas bajo de todos.
def evaluar_trivial(ds):
    y = ds["mora"].astype(int).to_numpy()
    grupos = ds["a_rutcliente"].to_numpy()
    X = ds.drop(columns=["mora"])
    filas, predichas = [], []
    cm = np.zeros((2, 2), dtype=int)
    for tr, te in hacer_splits(y, grupos):
        clf = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
        clf.fit(X.iloc[tr], y[tr])
        yp = clf.predict(X.iloc[te])
        sc = clf.predict_proba(X.iloc[te])[:, 1]
        predichas.append(int(yp[0]))
        filas.append(calcular_metricas(y[te], yp, sc))
        cm += confusion_matrix(y[te], yp, labels=[0, 1])
    out = resumir(filas, "Baseline trivial", cm)
    unicas = sorted(set(predichas))
    out["clase_predicha"] = ("mora(1)" if unicas == [1] else
                             "pagado(0)" if unicas == [0] else
                             "INESTABLE " + str(predichas))
    return out


# =============================================================================
#  PASO 6 - Los modelos y la validacion cruzada
# =============================================================================
# Comparo seis modelos. Los cuatro tabulares van aca; la CNN y la LSTM estan en
# el paso siguiente porque necesitan los datos en otro formato.
#
# Lo mas importante de esta seccion no son los modelos sino el SPLIT. Uso
# StratifiedGroupKFold agrupando por cliente. Si un mismo cliente tiene dos
# creditos y uno cae en entrenamiento y el otro en prueba, el modelo lo
# reconoce y las metricas salen infladas. Agrupando por RUT eso no puede pasar.
#
# Ademas, todo el preprocesamiento (imputar, escalar, one-hot) va DENTRO del
# pipeline, no antes. Si uno escala con la media de todo el dataset y despues
# parte, esa media ya vio los datos de prueba.
def hacer_splits(y, grupos, n_splits=5):
    n = min(n_splits, int(min(np.bincount(y))), len(np.unique(grupos)))
    n = max(2, n)
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEMILLA)
    return list(cv.split(np.zeros(len(y)), y, grupos))


def calcular_metricas(y_true, y_pred, scores=None):
    m = {"Accuracy": accuracy_score(y_true, y_pred),
         "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
         "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
         "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0)}
    # Las AUC solo tienen sentido si el fold tiene las dos clases
    if scores is not None and len(np.unique(y_true)) > 1:
        m["ROC_AUC"] = roc_auc_score(y_true, scores)
        m["PR_AUC"] = average_precision_score(y_true, scores)
    else:
        m["ROC_AUC"] = np.nan
        m["PR_AUC"] = np.nan
    return m


def resumir(filas, nombre, cm=None):
    m = pd.DataFrame(filas)
    out = {"modelo": nombre}
    for c in METRICAS:
        out[c] = m[c].mean()
        out[c + "_std"] = m[c].std()
    if cm is not None:
        out["TN"], out["FP"], out["FN"], out["TP"] = [int(v) for v in cm.ravel()]
    return out


def columnas(ds):
    win = [c for c in ds.columns if c.startswith("w_")]
    num = [c for c in EXO_NUM if c in ds.columns] + win
    cat = [c for c in EXO_CAT if c in ds.columns]
    return num, cat


def preprocesador(num, cat):
    """Imputar + escalar lo numerico, imputar + one-hot lo categorico.
    min_frequency=10 agrupa las categorias raras y evita cientos de columnas
    con dos o tres casos cada una."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore",
                                              min_frequency=10))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)],
                             remainder="drop")


def modelos_tabulares():
    # class_weight="balanced" y scale_pos_weight compensan el desbalance. Sin
    # eso los modelos se acomodan a responder "mora" casi siempre, que es
    # exactamente lo que hace el clasificador trivial.
    return {
        "Reg. Logística": LogisticRegression(max_iter=2000,
                                             class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400,
                                                class_weight="balanced",
                                                random_state=SEMILLA, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4,
                                 learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="logloss",
                                 random_state=SEMILLA, n_jobs=-1),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=SEMILLA),
    }


def puntajes(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    return None


def evaluar_modelo(modelo, ds):
    num, cat = columnas(ds)
    X = ds[num + cat]
    y = ds["mora"].astype(int).to_numpy()
    grupos = ds["a_rutcliente"].to_numpy()
    filas = []
    cm = np.zeros((2, 2), dtype=int)
    for tr, te in hacer_splits(y, grupos):
        pipe = Pipeline([("prep", preprocesador(num, cat)),
                         ("clf", clone(modelo))])
        pipe.fit(X.iloc[tr], y[tr])
        yp = pipe.predict(X.iloc[te])
        filas.append(calcular_metricas(y[te], yp, puntajes(pipe, X.iloc[te])))
        cm += confusion_matrix(y[te], yp, labels=[0, 1])
    return resumir(filas, None, cm)


# =============================================================================
#  PASO 7 - CNN y LSTM
# =============================================================================
# Los cuatro modelos anteriores reciben un vector de resumen por credito. Estas
# dos redes reciben otra cosa: la SECUENCIA cuota a cuota, con tres valores por
# paso (si se pago, cuanto atraso, si hubo atraso). La idea es que puedan captar
# el orden, que un resumen promediado pierde.
#
# Son redes hibridas: la rama de secuencia produce un vector de 32, y se le
# concatenan las variables de originacion antes de la capa final.
try:
    import torch
    import torch.nn as nn
    HAY_TORCH = True
    DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAY_TORCH = False
    DISPOSITIVO = None


def construir_secuencias(df, k, ids):
    obs = ventana_observada(df, k=k).set_index(["c_simulacion", "c_cuota"]).sort_index()
    presentes = set(obs.index.get_level_values(0))
    X = np.zeros((len(ids), k, 3), dtype="float32")
    for i, cid in enumerate(ids):
        if cid not in presentes:
            continue
        for cuota, fila in obs.loc[cid].iterrows():
            t = int(cuota) - 1
            if 0 <= t < k:
                X[i, t, 0] = fila["pagada_obs"]
                X[i, t, 1] = fila["atraso_efectivo"]
                X[i, t, 2] = fila["es_atraso"]
    return X


def construir_red(tipo, n_feat, k, n_exo):
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            ks = min(2, k)
            self.conv = nn.Sequential(
                nn.Conv1d(n_feat, 32, ks, padding="same"), nn.ReLU(),
                nn.Conv1d(32, 32, ks, padding="same"), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Linear(32 + n_exo, 16), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x, e=None):
            z = self.conv(x.transpose(1, 2)).flatten(1)
            if e is not None:
                z = torch.cat([z, e], dim=1)
            return self.head(z).squeeze(-1)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + n_exo, 16), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x, e=None):
            _, (h, _) = self.lstm(x)
            z = h[-1]
            if e is not None:
                z = torch.cat([z, e], dim=1)
            return self.head(z).squeeze(-1)

    return CNN1D() if tipo == "CNN" else LSTMNet()


def evaluar_red(tipo, df, ds, k, epochs=40, lr=1e-3):
    ids = ds.index.to_numpy()
    X = construir_secuencias(df, k, ids)
    y = ds["mora"].astype(int).to_numpy()
    grupos = ds["a_rutcliente"].to_numpy()
    num_exo = [c for c in EXO_NUM if c in ds.columns]
    cat_exo = [c for c in EXO_CAT if c in ds.columns]
    E = ds[num_exo + cat_exo]

    n_feat = X.shape[2]
    filas = []
    cm = np.zeros((2, 2), dtype=int)
    for tr, te in hacer_splits(y, grupos):
        # estandarizo la secuencia con la media y desviacion del TRAIN, no del total
        mu = X[tr].reshape(-1, n_feat).mean(0)
        sd = X[tr].reshape(-1, n_feat).std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, device=DISPOSITIVO)
        Xte = torch.tensor((X[te] - mu) / sd, device=DISPOSITIVO)
        ytr = torch.tensor(y[tr], dtype=torch.float32, device=DISPOSITIVO)

        prep = preprocesador(num_exo, cat_exo)
        Etr_np = prep.fit_transform(E.iloc[tr])
        Ete_np = prep.transform(E.iloc[te])
        Etr_np = (Etr_np.toarray() if hasattr(Etr_np, "toarray")
                  else np.asarray(Etr_np)).astype("float32")
        Ete_np = (Ete_np.toarray() if hasattr(Ete_np, "toarray")
                  else np.asarray(Ete_np)).astype("float32")
        Etr = torch.tensor(Etr_np, device=DISPOSITIVO)
        Ete = torch.tensor(Ete_np, device=DISPOSITIVO)

        net = construir_red(tipo, n_feat, k, Etr_np.shape[1]).to(DISPOSITIVO)
        pos = max(1, int((y[tr] == 1).sum()))
        neg = max(1, int((y[tr] == 0).sum()))
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], device=DISPOSITIVO))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            crit(net(Xtr, Etr), ytr).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            prob = torch.sigmoid(net(Xte, Ete)).cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        filas.append(calcular_metricas(y[te], pred, prob))
        cm += confusion_matrix(y[te], pred, labels=[0, 1])
    return resumir(filas, tipo, cm)


def evaluar_todos(ds, df=None, k=None):
    """Los seis modelos sobre un mismo dataset."""
    res = []
    for nombre, mod in modelos_tabulares().items():
        r = evaluar_modelo(mod, ds)
        r["modelo"] = nombre
        res.append(r)
        print("      %-16s F1 %.3f" % (nombre, r["F1"]))
    if HAY_TORCH and df is not None and k is not None:
        for tipo in ("CNN", "LSTM"):
            r = evaluar_red(tipo, df, ds, k)
            res.append(r)
            print("      %-16s F1 %.3f" % (tipo, r["F1"]))
    return pd.DataFrame(res)


# =============================================================================
#  PASO 8 - Los cuatro objetivos
# =============================================================================
def main():
    os.makedirs(SALIDA, exist_ok=True)

    df = cargar()
    obs_max = marcar_observadas(df)
    exo, target = exogenas_y_target(df)
    DS, ds_techo = armar_datasets(df, exo, target, obs_max)

    if not HAY_TORCH:
        print("\n  AVISO: no encontre PyTorch, asi que salto la CNN y la LSTM.")
        print("  Los otros cuatro modelos corren igual.")
    else:
        print("\n  PyTorch disponible (%s): corro los seis modelos." % DISPOSITIVO)

    # ---- distribucion de clases, que es el contexto de todo lo demas --------
    filas = [{"ambito": "Techo (hasta penultima cuota)", "n": len(ds_techo),
              "pct_mora": round(ds_techo["mora"].mean() * 100, 1)}]
    for k in VENTANAS:
        filas.append({"ambito": "Ventana k=%d" % k, "n": len(DS[k]),
                      "pct_mora": round(DS[k]["mora"].mean() * 100, 1)})
    pd.DataFrame(filas).to_csv(
        os.path.join(SALIDA, "distribucion_clases.csv"), index=False)

    # ---- OBJETIVO 1: el techo ----------------------------------------------
    # Cuanto se puede saber usando TODO el historial hasta la penultima cuota.
    # No es un modelo util -en la penultima cuota ya casi se sabe el desenlace-
    # sino la referencia superior contra la que comparar las ventanas cortas.
    titulo("OBJETIVO 1 - El techo: todo el historial hasta la penultima cuota")
    r1 = evaluar_todos(ds_techo)
    t1 = evaluar_trivial(ds_techo)
    print("      %-16s F1 %.3f   (predice %s)"
          % ("TRIVIAL", t1["F1"], t1["clase_predicha"]))
    r1.to_csv(os.path.join(SALIDA, "obj1_techo.csv"), index=False)

    # ---- OBJETIVO 2: la ventana de 5 cuotas --------------------------------
    titulo("OBJETIVO 2 - Seis modelos con una ventana de 5 cuotas")
    r2 = evaluar_todos(DS[5], df, 5)
    t2 = evaluar_trivial(DS[5])
    print("      %-16s F1 %.3f   (predice %s)"
          % ("TRIVIAL", t2["F1"], t2["clase_predicha"]))
    r2.to_csv(os.path.join(SALIDA, "obj2_ventana_k5.csv"), index=False)

    # por segmento de plazo: aca es donde aparece el hallazgo mas interesante
    filas = []
    for grp in GRUPOS_PLAZO:
        sub = DS[5][DS[5]["grupo_plazo"] == grp]
        if len(sub) < 30 or sub["mora"].nunique() < 2:
            continue
        tg = evaluar_trivial(sub)
        rg = evaluar_modelo(modelos_tabulares()["XGBoost"], sub)
        filas.append({"grupo_plazo": grp, "n": len(sub),
                      "pct_mora": round(sub["mora"].mean() * 100, 1),
                      "F1_trivial": round(tg["F1"], 3),
                      "F1_xgboost": round(rg["F1"], 3),
                      "margen": round(rg["F1"] - tg["F1"], 3),
                      "clase_predicha_trivial": tg["clase_predicha"]})
        print("    plazo %-3d n=%-4d %%mora=%4.1f  trivial %.3f  XGBoost %.3f  margen %+.3f"
              % (grp, len(sub), sub["mora"].mean() * 100,
                 tg["F1"], rg["F1"], rg["F1"] - tg["F1"]))
    pd.DataFrame(filas).to_csv(
        os.path.join(SALIDA, "obj2_por_plazo.csv"), index=False)

    # ---- OBJETIVO 3: todas las ventanas ------------------------------------
    titulo("OBJETIVO 3 - Como cambia el rendimiento al variar la ventana")
    todos, triviales = [], []
    for k in VENTANAS:
        print("    --- k = %d (%d creditos, %.1f%% mora) ---"
              % (k, len(DS[k]), DS[k]["mora"].mean() * 100))
        r = evaluar_todos(DS[k], df, k)
        r["ventana_k"] = k
        todos.append(r)
        t = evaluar_trivial(DS[k])
        t["ventana_k"] = k
        t["n"] = len(DS[k])
        t["pct_mora"] = round(DS[k]["mora"].mean() * 100, 1)
        triviales.append(t)
        print("      %-16s F1 %.3f" % ("TRIVIAL", t["F1"]))
    obj3 = pd.concat(todos, ignore_index=True)
    obj3.to_csv(os.path.join(SALIDA, "obj3_todas_las_ventanas.csv"), index=False)
    pd.DataFrame(triviales).to_csv(
        os.path.join(SALIDA, "baseline_trivial.csv"), index=False)

    # ---- OBJETIVO 4: la cohorte comun --------------------------------------
    # Este es el objetivo que de verdad responde la pregunta. Los tres anteriores
    # comparan ventanas sobre poblaciones distintas, asi que mezclan el efecto de
    # la ventana con el de la poblacion. Aca fijo los mismos creditos -los que
    # sobreviven a la ventana mas larga- y vuelvo a medir. Recien ahi la
    # comparacion entre k = 1 y k = 10 es una comparacion honesta.
    titulo("OBJETIVO 4 - Cohorte comun: misma poblacion en todas las ventanas")
    comun = DS[max(VENTANAS)].index
    print("  Fijo los %d creditos que sobreviven a k = %d\n"
          % (len(comun), max(VENTANAS)))
    filas = []
    for k in VENTANAS:
        sub = DS[k].loc[DS[k].index.intersection(comun)]
        print("    --- k = %d (%d creditos, %.1f%% mora) ---"
              % (k, len(sub), sub["mora"].mean() * 100))
        r = evaluar_todos(sub, df, k)
        r["ventana_k"] = k
        r["n_comun"] = len(sub)
        filas.append(r)
        t = evaluar_trivial(sub)
        print("      %-16s F1 %.3f" % ("TRIVIAL", t["F1"]))
    obj4 = pd.concat(filas, ignore_index=True)
    obj4.to_csv(os.path.join(SALIDA, "obj4_cohorte_comun.csv"), index=False)

    # ---- el resultado que resume todo --------------------------------------
    titulo("RESUMEN")
    v = obj3.groupby("ventana_k")["F1"].max()
    c = obj4.groupby("ventana_k")["F1"].max()
    print("  El mejor F1 de cada ventana, de dos maneras:\n")
    print("    %-6s %-22s %-22s" % ("k", "poblacion variable", "cohorte fija"))
    for k in VENTANAS:
        print("    %-6d %-22.3f %-22.3f" % (k, v[k], c[k]))
    print()
    print("  Sobre poblacion variable, pasar de 1 a %d cuotas parece valer"
          % max(VENTANAS))
    print("  solo %+.3f de F1. Sobre la misma poblacion, vale %+.3f."
          % (v[max(VENTANAS)] - v[1], c[max(VENTANAS)] - c[1]))
    print()
    print("  Esa diferencia es todo el punto del trabajo: el F1 de k = 1 se ve")
    print("  alto porque su base de mora es alta, no porque el modelo prediga")
    print("  bien. Al fijar la poblacion, el aporte real de observar mas cuotas")
    print("  aparece completo.")
    print("\n  Tablas escritas en: %s" % SALIDA)


if __name__ == "__main__":
    main()
