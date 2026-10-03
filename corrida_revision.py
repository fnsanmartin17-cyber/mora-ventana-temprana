# -*- coding: utf-8 -*-
"""
corrida_revision.py
===================
Corrida complementaria para la revision del articulo. NO modifica el script
original: lo importa y reutiliza sus funciones (carga, ventanas, variables,
modelos), asi que las variables de comportamiento se calculan exactamente igual
que en el articulo.

Uso (en la misma carpeta que analisis_mora_temprana.py):

    DATOS_CSV=/ruta/a/la/base_real.csv python corrida_revision.py

Opciones por variable de entorno:
    BLOQUES=0,1,2,3,4   bloques a correr (por defecto todos)
    REPS=5              repeticiones de la validacion cruzada (semillas distintas)
    SIN_REDES=1         omite CNN y LSTM (para una corrida rapida de prueba)

Salida: carpeta resultados_revision/
    - Archivos agregados (*.csv, resumen.txt): no contienen RUT ni datos por credito.
      Son los que hay que enviar.
    - privado_oof_predicciones.csv: predicciones fuera de fold por credito. Queda
      local; solo compartirla si el acuerdo de confidencialidad lo permite.

Bloques:
    0  Diagnostico de la variable objetivo (M1) y del prepago (M2). Minutos.
    1  Objetivo original, cohorte comun: ablacion del prepago, solo-originacion,
       solo-ventana, XGBoost con/sin ponderacion, metricas completas (M2-M6).
    2  Objetivo corregido: default = 90+ dias de atraso dentro de 12 (y 6) meses
       despues del punto de observacion, solo creditos activos en ese punto (M1).
    3  Objetivo original restringido a creditos ya vencidos (M1, alternativa).
    4  Sensibilidad del objetivo corregido (60 dias; horizonte 6 meses).
    5  Prepago dentro de 6 meses como evento propio (riesgo competitivo).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import analisis_mora_temprana as base
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix)
from xgboost import XGBClassifier
from scipy import stats

SALIDA = os.path.join(base.AQUI, "resultados_revision")
os.makedirs(SALIDA, exist_ok=True)

BLOQUES = [int(b) for b in os.environ.get("BLOQUES", "0,1,2,3,4,5").split(",")]
REPS = int(os.environ.get("REPS", "5"))
SEMILLAS = [42, 7, 123, 2024, 99, 314, 2718, 1, 55, 808][:REPS]
REDES = base.HAY_TORCH and os.environ.get("SIN_REDES", "0") != "1"
VENTANAS = base.VENTANAS
N_INNER = 3             # folds internos para elegir el umbral (sin mirar el test)

LOG = open(os.path.join(SALIDA, "resumen.txt"), "w", encoding="utf-8")


def log(*a):
    txt = " ".join(str(x) for x in a)
    print(txt)
    LOG.write(txt + "\n")
    LOG.flush()


# =============================================================================
#  Preparacion comun
# =============================================================================
def preparar():
    df = base.cargar()
    obs_max = base.marcar_observadas(df)          # agrega df["observada"]
    exo, target = base.exogenas_y_target(df)
    corte = df.loc[df["pagada"] == 1, "d_pago"].max()
    return df, obs_max, exo, target, corte


def fecha_default(df, corte, D):
    """Fecha en que cada credito alcanza por primera vez D dias de atraso en
    alguna cuota (NaT si nunca). Una cuota pagada con dias_atraso >= D o una
    cuota impaga cuyo vencimiento + D ya ocurrio antes del corte generan el
    evento en la fecha vencimiento + D."""
    venc = df["d_vencimiento"]
    ev_pag = (df["pagada"] == 1) & (df["dias_atraso"] >= D)
    ev_imp = (df["pagada"] == 0) & (venc + pd.Timedelta(days=D) <= corte)
    fecha = (venc + pd.Timedelta(days=D)).where(ev_pag | ev_imp)
    return fecha.groupby(df["c_simulacion"]).min()


def fecha_cuota(df, k):
    return df[df["c_cuota"] == k].set_index("c_simulacion")["d_vencimiento"]


def fecha_pago_total(df, target):
    """Fecha en que el credito quedo totalmente pagado (NaT si no lo esta):
    el ultimo pago de los creditos con mora = 0."""
    pag = df[df["pagada"] == 1]
    ult = pag.groupby("c_simulacion")["d_pago"].max()
    return ult.where(target["mora"].reindex(ult.index) == 0)


def activos_en(df, target, k):
    """Creditos que en t_k (vencimiento de la cuota k) todavia NO estaban
    totalmente pagados. Un credito ya liquidado en t_k tiene su desenlace
    decidido: incluirlo es predecir algo que ya ocurrio."""
    tk = fecha_cuota(df, k)
    fp = fecha_pago_total(df, target).reindex(tk.index)
    return tk.index[~(fp <= tk)]


def objetivo_corregido(df, obs_max, corte, k, D=90, H=365, target=None,
                       evento="default"):
    """Etiqueta: 1 si el credito alcanza D dias de atraso (evento='default') o
    queda totalmente pagado de forma anticipada (evento='prepago') dentro de
    los H dias siguientes al punto de observacion t_k (vencimiento de la cuota k).
    Elegibles: creditos con k cuotas observadas, cuya ventana de desempeno
    completa (t_k + H) ya ocurrio antes del corte, que seguian activos en t_k
    (no liquidados) y que NO estaban ya en default en t_k."""
    fd = fecha_default(df, corte, D)
    tk = fecha_cuota(df, k)
    elig = obs_max[obs_max >= k].index.intersection(tk.index)
    if target is not None:
        elig = elig.intersection(activos_en(df, target, k))
    tk = tk.loc[elig]
    fd = fd.reindex(elig)
    ok = (tk + pd.Timedelta(days=H) <= corte) & ~(fd <= tk)
    if evento == "prepago":
        fp = fecha_pago_total(df, target).reindex(elig)
        y = ((fp > tk) & (fp <= tk + pd.Timedelta(days=H))).astype(int)
    else:
        y = ((fd > tk) & (fd <= tk + pd.Timedelta(days=H))).astype(int)
    return y[ok]


def armar(df, exo, ids, y, k, conjunto="completo"):
    """Dataset con el mismo formato del script original (columna 'mora')."""
    ids = pd.Index(ids)
    if k == 0 or conjunto == "solo_originacion":
        ds = exo.loc[exo.index.intersection(ids)].copy()
    else:
        fw = base.features_ventana(df, k=k)
        if conjunto == "sin_anticipo":
            fw = fw.drop(columns=["w_anticipo_medio"])
        ds = exo.join(fw, how="inner").loc[lambda d: d.index.isin(ids)].copy()
        if conjunto == "solo_ventana":
            ds = ds[[c for c in ds.columns if c.startswith("w_")] +
                    ["a_rutcliente", "grupo_plazo"]].copy()
    ds["mora"] = y.reindex(ds.index)
    ds = ds.dropna(subset=["mora"])
    ds["mora"] = ds["mora"].astype(int)
    return ds


# =============================================================================
#  Modelos: cada uno es una funcion (ds, tr, te, semilla) -> puntajes del test
# =============================================================================
def columnas(ds):
    num, cat = base.columnas(ds)
    return num, cat


def tabular(modelo, spw=False):
    def f(ds, tr, te, semilla, df=None, k=None):
        num, cat = columnas(ds)
        X = ds[num + cat]
        y = ds["mora"].to_numpy().astype(int)
        m = clone(modelo)
        if "random_state" in m.get_params():
            m.set_params(random_state=semilla)
        if spw:
            pos, neg = max(1, (y[tr] == 1).sum()), max(1, (y[tr] == 0).sum())
            m.set_params(scale_pos_weight=neg / pos)
        pipe = Pipeline([("prep", base.preprocesador(num, cat)), ("clf", m)])
        pipe.fit(X.iloc[tr], y[tr])
        s = base.puntajes(pipe, X.iloc[te])
        es_prob = hasattr(pipe, "predict_proba")
        return np.asarray(s, dtype=float), es_prob
    return f


def red(tipo):
    def f(ds, tr, te, semilla, df=None, k=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(semilla)
        np.random.seed(semilla)
        ids = ds.index.to_numpy()
        if k not in SEQ_CACHE:                      # secuencias de TODOS los creditos, una vez por k
            todos = df["c_simulacion"].unique()
            SEQ_CACHE[k] = (pd.Index(todos), base.construir_secuencias(df, k, todos))
        idx_all, X_all = SEQ_CACHE[k]
        X = X_all[idx_all.get_indexer(ids)]
        if "w_anticipo_medio" not in ds.columns:     # ablacion del prepago tambien en la secuencia
            X = X.copy()
            X[:, :, 1] = np.clip(X[:, :, 1], 0, None)
        y = ds["mora"].to_numpy().astype(int)
        num_exo = [c for c in base.EXO_NUM if c in ds.columns]
        cat_exo = [c for c in base.EXO_CAT if c in ds.columns]
        E = ds[num_exo + cat_exo]
        nf = X.shape[2]
        dev = base.DISPOSITIVO
        mu = X[tr].reshape(-1, nf).mean(0)
        sd = X[tr].reshape(-1, nf).std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, device=dev)
        Xte = torch.tensor((X[te] - mu) / sd, device=dev)
        ytr = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        prep = base.preprocesador(num_exo, cat_exo)
        a = prep.fit_transform(E.iloc[tr]); b = prep.transform(E.iloc[te])
        a = (a.toarray() if hasattr(a, "toarray") else np.asarray(a)).astype("float32")
        b = (b.toarray() if hasattr(b, "toarray") else np.asarray(b)).astype("float32")
        Etr = torch.tensor(a, device=dev); Ete = torch.tensor(b, device=dev)
        net = base.construir_red(tipo, nf, k, a.shape[1]).to(dev)
        pos, neg = max(1, (y[tr] == 1).sum()), max(1, (y[tr] == 0).sum())
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=dev))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        net.train()
        for _ in range(40):
            opt.zero_grad(); crit(net(Xtr, Etr), ytr).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xte, Ete)).cpu().numpy()
        return p.astype(float), True
    return f


SEQ_CACHE = {}


def catalogo(con_redes=True, solo=None):
    t = base.modelos_tabulares()
    cat = {"Reg. Logistica": tabular(t["Reg. Logística"]),
           "Random Forest": tabular(t["Random Forest"]),
           "XGBoost": tabular(t["XGBoost"]),                     # tal como el articulo
           "XGBoost (spw)": tabular(t["XGBoost"], spw=True),     # con ponderacion
           "SVM": tabular(t["SVM"])}
    if con_redes and REDES:
        cat["CNN"] = red("CNN")
        cat["LSTM"] = red("LSTM")
    if solo:
        cat = {k: v for k, v in cat.items() if k in solo}
    return cat


# =============================================================================
#  Evaluacion
# =============================================================================
def splits(y, g, semilla, n=5):
    n = max(2, min(n, int(np.bincount(y).min()), len(np.unique(g))))
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla)
    return list(cv.split(np.zeros(len(y)), y, g))


def umbral_interno(fn, ds, tr, semilla, df, k):
    """Umbral que maximiza MCC en validacion cruzada interna SOLO sobre train."""
    y = ds["mora"].to_numpy().astype(int)
    g = ds["a_rutcliente"].to_numpy()
    sub = ds.iloc[tr]
    ysub, gsub = y[tr], g[tr]
    oof = np.full(len(tr), np.nan)
    for a, b in splits(ysub, gsub, semilla + 1, N_INNER):
        s, _ = fn(sub, a, b, semilla, df, k)
        oof[b] = s
    m = ~np.isnan(oof)
    cand = np.unique(np.quantile(oof[m], np.linspace(0.02, 0.98, 97)))
    mejores = [(matthews_corrcoef(ysub[m], (oof[m] >= c).astype(int)), c) for c in cand]
    return max(mejores)[1]


def umbral_defecto(es_prob):
    return 0.5 if es_prob else 0.0


def metricas(y, s, thr, es_prob):
    p = (s >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = {"Accuracy": accuracy_score(y, p),
           "F1_pos": f1_score(y, p, pos_label=1, zero_division=0),
           "F1_neg": f1_score(y, p, pos_label=0, zero_division=0),
           "Recall": recall_score(y, p, zero_division=0),
           "Precision": precision_score(y, p, zero_division=0),
           "Especificidad": tn / max(1, tn + fp),
           "BalAcc": balanced_accuracy_score(y, p),
           "MCC": matthews_corrcoef(y, p),
           "TN": tn, "FP": fp, "FN": fn_, "TP": tp}
    out["MacroF1"] = (out["F1_pos"] + out["F1_neg"]) / 2
    if len(np.unique(y)) > 1:
        out["ROC_AUC"] = roc_auc_score(y, s)
        out["PR_AUC_pos"] = average_precision_score(y, s)
        out["PR_AUC_neg"] = average_precision_score(1 - y, -s)
        # capacidad de gestion: de los que caen en mora, cuantos quedan en el
        # 10% / 20% de mayor puntaje
        orden = np.argsort(-s)
        for q in (0.1, 0.2):
            top = orden[:max(1, int(round(q * len(s))))]
            out["Recall@%d%%" % int(q * 100)] = y[top].sum() / max(1, y.sum())
            out["Lift@%d%%" % int(q * 100)] = y[top].mean() / max(1e-9, y.mean())
    out["Brier"] = brier_score_loss(y, np.clip(s, 0, 1)) if es_prob else np.nan
    return out


def evaluar(nombre, fn, ds, df=None, k=None, etiqueta="", oof_rows=None):
    y = ds["mora"].to_numpy().astype(int)
    g = ds["a_rutcliente"].to_numpy()
    filas = []
    for semilla in SEMILLAS:
        for f, (tr, te) in enumerate(splits(y, g, semilla)):
            s, es_prob = fn(ds, tr, te, semilla, df, k)
            thr_d = umbral_defecto(es_prob)
            thr_t = umbral_interno(fn, ds, tr, semilla, df, k)
            md = metricas(y[te], s, thr_d, es_prob)
            mt = metricas(y[te], s, thr_t, es_prob)
            fila = {"semilla": semilla, "fold": f, "n_test": len(te),
                    "n_train": len(tr)}
            fila.update({k2: v for k2, v in md.items()})
            fila.update({k2 + "_umbralMCC": v for k2, v in mt.items()
                         if k2 in ("F1_pos", "F1_neg", "MacroF1", "BalAcc", "MCC",
                                   "Recall", "Especificidad", "Precision")})
            filas.append(fila)
            if oof_rows is not None:
                oof_rows.append(pd.DataFrame({
                    "experimento": etiqueta, "modelo": nombre, "semilla": semilla,
                    "fold": f, "c_simulacion": ds.index.to_numpy()[te],
                    "grupo_plazo": ds["grupo_plazo"].to_numpy()[te],
                    "y": y[te], "score": s, "umbral_defecto": thr_d,
                    "umbral_mcc": thr_t}))
    return pd.DataFrame(filas)


def trivial(y):
    p = y.mean()
    maj = 1 if p >= 0.5 else 0
    pm = p if maj == 1 else 1 - p
    return {"prevalencia_pos": p, "clase_mayoritaria": maj,
            "F1_pos_trivial": 2 * p / (1 + p) if maj == 1 else 0.0,
            "F1_neg_trivial": 0.0 if maj == 1 else 2 * pm / (1 + pm),
            "PR_AUC_pos_azar": p, "ROC_AUC_azar": 0.5, "MCC_trivial": 0.0,
            "BalAcc_trivial": 0.5}


def resumir(res, cols_extra):
    num = res.select_dtypes("number").drop(columns=["semilla", "fold"], errors="ignore")
    m = num.mean().add_suffix("")
    s = num.std().add_suffix("_sd")
    out = pd.concat([m, s]).to_dict()
    out.update(cols_extra)
    return out


def nadeau_bengio(a, b, n_train, n_test):
    """t test corregido (Nadeau & Bengio 2003) sobre diferencias por fold."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return np.nan, np.nan
    var = np.var(d, ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def correr(experimento, ds, df, k, modelos, filas_res, filas_fold, oof_rows,
           extra=None):
    extra = extra or {}
    y = ds["mora"].to_numpy().astype(int)
    tv = trivial(y)
    log("    [%s] k=%s n=%d  %%pos=%.1f" % (experimento, k, len(ds), 100 * y.mean()))
    por_modelo = {}
    for nombre, fn in modelos.items():
        t0 = time.time()
        r = evaluar(nombre, fn, ds, df, k, "%s|k=%s" % (experimento, k), oof_rows)
        r["experimento"], r["k"], r["modelo"] = experimento, k, nombre
        filas_fold.append(r)
        por_modelo[nombre] = r
        info = {"experimento": experimento, "k": k, "modelo": nombre, "n": len(ds)}
        info.update(extra); info.update(tv)
        filas_res.append(resumir(r, info))
        log("      %-15s AUC %.3f  PR-AUC %.3f  MCC %.3f  F1 %.3f  (%.0fs)"
            % (nombre, r["ROC_AUC"].mean(), r["PR_AUC_pos"].mean(),
               r["MCC_umbralMCC"].mean(), r["F1_pos"].mean(), time.time() - t0))
    # tests pareados contra el mejor por AUC
    if len(por_modelo) > 1:
        mejor = max(por_modelo, key=lambda m: por_modelo[m]["ROC_AUC"].mean())
        for nombre, r in por_modelo.items():
            if nombre == mejor:
                continue
            for met in ("ROC_AUC", "MCC_umbralMCC", "F1_pos"):
                t, p = nadeau_bengio(por_modelo[mejor][met], r[met],
                                     r["n_train"].mean(), r["n_test"].mean())
                TESTS.append({"experimento": experimento, "k": k, "metrica": met,
                              "mejor": mejor, "vs": nombre,
                              "dif_media": por_modelo[mejor][met].mean() - r[met].mean(),
                              "t_corregido": t, "p_valor": p})


TESTS = []


def guardar(nombre, filas_res, filas_fold):
    if filas_res:
        pd.DataFrame(filas_res).to_csv(os.path.join(SALIDA, nombre + ".csv"), index=False)
    if filas_fold:
        f = pd.concat(filas_fold, ignore_index=True)
        f.to_csv(os.path.join(SALIDA, nombre + "_por_fold.csv"), index=False)


def por_plazo(oof, nombre):
    """Metricas por grupo de plazo a partir de las predicciones fuera de fold."""
    filas = []
    for (exp, mod, sem, gp), d in oof.groupby(["experimento", "modelo", "semilla",
                                               "grupo_plazo"]):
        y = d["y"].to_numpy(); s = d["score"].to_numpy()
        if len(y) < 20 or len(np.unique(y)) < 2:
            continue
        mt = metricas(y, s, d["umbral_mcc"].to_numpy(), True)
        md = metricas(y, s, d["umbral_defecto"].to_numpy(), True)
        filas.append({"experimento": exp, "modelo": mod, "semilla": sem,
                      "grupo_plazo": gp, "n": len(y), "pct_pos": 100 * y.mean(),
                      "ROC_AUC": mt["ROC_AUC"], "PR_AUC_pos": mt["PR_AUC_pos"],
                      "MCC_umbralMCC": mt["MCC"], "BalAcc_umbralMCC": mt["BalAcc"],
                      "F1_pos": md["F1_pos"],
                      "F1_pos_trivial": trivial(y)["F1_pos_trivial"]})
    if filas:
        r = (pd.DataFrame(filas).groupby(["experimento", "modelo", "grupo_plazo"])
             .mean(numeric_only=True).drop(columns="semilla").reset_index())
        r.to_csv(os.path.join(SALIDA, nombre + "_por_plazo.csv"), index=False)


# =============================================================================
#  BLOQUE 0 - Diagnostico
# =============================================================================
def bloque0(df, obs_max, exo, target, corte):
    base.titulo("BLOQUE 0 - Diagnostico de la variable objetivo y del prepago")
    c = df.groupby("c_simulacion").first()
    mora = target["mora"]
    ult = df.groupby("c_simulacion")["d_vencimiento"].max()
    ini = df[df["c_cuota"] == 1].set_index("c_simulacion")["d_vencimiento"]
    pag = df[df["pagada"] == 1]
    impaga_vencida = (df[(df["pagada"] == 0) & (df["d_vencimiento"] <= corte)]
                      .groupby("c_simulacion").size().reindex(c.index).fillna(0))
    max_atraso_real = pag.groupby("c_simulacion")["dias_atraso"].max().reindex(c.index)
    prepago60 = (pag["dias_atraso"] <= -60).groupby(pag["c_simulacion"]).any() \
        .reindex(c.index).fillna(False)
    prepago30 = (pag["dias_atraso"] <= -30).groupby(pag["c_simulacion"]).any() \
        .reindex(c.index).fillna(False)
    vencido = ult <= corte
    fd90 = fecha_default(df, corte, 90).reindex(c.index)

    d = pd.DataFrame({"mora": mora, "plazo": c["total_cuotas"],
                      "impagas_vencidas_al_corte": impaga_vencida,
                      "credito_vencido_al_corte": vencido,
                      "prepago_le_-60": prepago60, "prepago_le_-30": prepago30,
                      "alguna_vez_90dpd": fd90.notna(),
                      "obs_max": obs_max.reindex(c.index),
                      "trimestre_origen": ini.reindex(c.index).dt.to_period("Q").astype(str)})
    if "estado_pago_credito" in c.columns:
        d["estado_pago_credito"] = c["estado_pago_credito"]

    log("  Fecha de corte: %s" % corte.date())
    log("  Creditos: %d   Clientes: %d   Creditos por cliente (max): %d"
        % (len(c), c["a_rutcliente"].nunique(),
           c.groupby("a_rutcliente").size().max()))

    # --- La prueba clave de M1 ---
    al_dia = (d["mora"] == 1) & (d["impagas_vencidas_al_corte"] == 0)
    log("\n  PRUEBA CLAVE M1: creditos con mora=1 SIN ninguna cuota vencida impaga "
        "al corte: %d de %d (%.1f%%)" % (al_dia.sum(), (d["mora"] == 1).sum(),
                                         100 * al_dia.mean() / max(1e-9, d['mora'].mean())))
    log("  Creditos con mora=1 y sin cuotas vencidas impagas, que ademas nunca "
        "tuvieron 90+ dias de atraso: %d" % (al_dia & ~d["alguna_vez_90dpd"]).sum())

    tablas = {}
    if "estado_pago_credito" in d:
        tablas["estado_x_mora"] = pd.crosstab(d["estado_pago_credito"], d["mora"], margins=True)
    tablas["impagas_vencidas_x_mora"] = pd.crosstab(d["impagas_vencidas_al_corte"] > 0,
                                                    d["mora"], margins=True)
    tablas["vencido_x_mora"] = pd.crosstab(d["credito_vencido_al_corte"], d["mora"], margins=True)
    tablas["plazo_x_mora"] = pd.crosstab(d["plazo"], d["mora"], margins=True)
    tablas["trimestre_origen_x_mora"] = pd.crosstab(d["trimestre_origen"], d["mora"], margins=True)
    tablas["prepago60_x_mora"] = pd.crosstab(d["prepago_le_-60"], d["mora"], margins=True)
    tablas["prepago30_x_mora"] = pd.crosstab(d["prepago_le_-30"], d["mora"], margins=True)
    tablas["90dpd_x_mora"] = pd.crosstab(d["alguna_vez_90dpd"], d["mora"], margins=True)
    tablas["obsmax_lt10_x_mora_x_impagas"] = pd.crosstab(
        [d["obs_max"] < 10, d["impagas_vencidas_al_corte"] > 0], d["mora"], margins=True)
    for n, t in tablas.items():
        log("\n  --- %s ---\n%s" % (n, t.to_string()))
        t.to_csv(os.path.join(SALIDA, "diag_%s.csv" % n))

    # prepago dentro de cada ventana (M2): en que cuota aparece el -60
    filas = []
    for k in VENTANAS:
        w = df[(df["c_cuota"] <= k) & (df["pagada"] == 1)]
        f60 = (w["dias_atraso"] <= -60).groupby(w["c_simulacion"]).any()
        elig = obs_max[obs_max >= k].index
        f60 = f60.reindex(elig).fillna(False)
        mm = mora.reindex(elig)
        filas.append({"k": k, "n": len(elig), "con_-60_en_ventana": int(f60.sum()),
                      "de_ellos_mora0": int((f60 & (mm == 0)).sum()),
                      "de_ellos_mora1": int((f60 & (mm == 1)).sum())})
    t = pd.DataFrame(filas)
    log("\n  --- Prepago (-60) observado dentro de la ventana ---\n%s" % t.to_string(index=False))
    t.to_csv(os.path.join(SALIDA, "diag_prepago_en_ventana.csv"), index=False)

    # tamano de muestra con el objetivo corregido
    filas = []
    for ev, D, H in (("default", 90, 365), ("default", 90, 180),
                     ("default", 60, 180), ("prepago", 90, 180)):
        for k in VENTANAS:
            y = objetivo_corregido(df, obs_max, corte, k, D, H, target, ev)
            filas.append({"evento": ev, "D_dias": D, "H_dias": H, "k": k,
                          "n_elegibles": len(y), "n_evento": int(y.sum()),
                          "pct_evento": round(100 * y.mean(), 1) if len(y) else np.nan})
    t = pd.DataFrame(filas)
    log("\n  --- Tamano de muestra con el objetivo corregido ---\n%s" % t.to_string(index=False))
    t.to_csv(os.path.join(SALIDA, "diag_objetivo_corregido_n.csv"), index=False)

    # creditos ya liquidados en el punto de observacion t_k (fuga de desenlace)
    filas = []
    for k in VENTANAS:
        elig = obs_max[obs_max >= k].index
        act = activos_en(df, target, k).intersection(elig)
        cerr = elig.difference(act)
        filas.append({"k": k, "n": len(elig), "ya_liquidados_en_tk": len(cerr),
                      "de_ellos_mora0": int((mora.reindex(cerr) == 0).sum()),
                      "total_mora0_en_ventana": int((mora.reindex(elig) == 0).sum())})
    t = pd.DataFrame(filas)
    log("\n  --- Creditos YA liquidados en el punto de observacion t_k ---\n%s"
        % t.to_string(index=False))
    t.to_csv(os.path.join(SALIDA, "diag_liquidados_en_tk.csv"), index=False)

    mat = d[d["credito_vencido_al_corte"]]
    log("\n  Creditos ya vencidos al corte: %d (mora %.1f%%)"
        % (len(mat), 100 * mat["mora"].mean() if len(mat) else np.nan))
    for k in VENTANAS:
        log("    con >= %d cuotas observadas: %d" % (k, (mat["obs_max"] >= k).sum()))


# =============================================================================
#  BLOQUE 1 - Objetivo original, cohorte comun, ablaciones
# =============================================================================
def bloque1(df, obs_max, exo, target, corte):
    base.titulo("BLOQUE 1 - Objetivo original, cohorte comun")
    y = target["mora"]
    comun = obs_max[obs_max >= max(VENTANAS)].index
    res, fold, oof = [], [], []
    mods = catalogo()
    ds0 = armar(df, exo, comun, y, 0)
    correr("orig_comun_solo_originacion", ds0, df, 0,
           catalogo(con_redes=False), res, fold, oof)
    for conjunto in ("completo", "sin_anticipo", "solo_ventana"):
        for k in VENTANAS:
            ds = armar(df, exo, comun, y, k, conjunto)
            m = mods if conjunto != "solo_ventana" else catalogo(con_redes=False)
            correr("orig_comun_" + conjunto, ds, df, k, m, res, fold, oof)
    # misma cohorte, pero solo creditos que seguian activos en t_10
    act = activos_en(df, target, max(VENTANAS)).intersection(comun)
    log("    Cohorte comun activa en t_%d: %d creditos" % (max(VENTANAS), len(act)))
    for k in VENTANAS:
        ds = armar(df, exo, act, y, k, "completo")
        correr("orig_comun_activos_t10", ds, df, k, mods, res, fold, oof)
    guardar("b1_original_cohorte_comun", res, fold)
    o = pd.concat(oof, ignore_index=True)
    o.to_csv(os.path.join(SALIDA, "privado_oof_b1.csv"), index=False)
    por_plazo(o, "b1_original_cohorte_comun")


# =============================================================================
#  BLOQUE 2 - Objetivo corregido (90+ dias en 12 meses)
# =============================================================================
def bloque2(df, obs_max, exo, target, corte, D=90, H=365, nombre="b2_corregido_90d_12m",
            modelos=None, conjuntos=("completo", "sin_anticipo"), evento="default"):
    base.titulo("BLOQUE - Objetivo corregido (%s): %d+ dias dentro de %d dias"
                % (evento, D, H))
    res, fold, oof = [], [], []
    mods = modelos or catalogo()
    ys = {k: objetivo_corregido(df, obs_max, corte, k, D, H, target, evento)
          for k in VENTANAS}
    # poblacion variable
    for k in VENTANAS:
        y = ys[k]
        if len(y) < 100 or y.nunique() < 2 or min(y.sum(), (1 - y).sum()) < 10:
            log("    k=%d: muestra insuficiente (n=%d, default=%d)" % (k, len(y), y.sum()))
            continue
        for conjunto in conjuntos:
            ds = armar(df, exo, y.index, y, k, conjunto)
            correr("corr_var_" + conjunto, ds, df, k, mods, res, fold, oof,
                   {"D": D, "H": H})
        if k == min(VENTANAS):
            ds0 = armar(df, exo, y.index, y, 0)
            correr("corr_var_solo_originacion", ds0, df, 0,
                   {m: f for m, f in mods.items() if m not in ("CNN", "LSTM")},
                   res, fold, oof, {"D": D, "H": H})
    # cohorte comun: elegibles en la ventana mas larga que tenga muestra
    kmax = max([k for k in VENTANAS if len(ys[k]) >= 100 and ys[k].nunique() == 2]
               or [None], default=None)
    if kmax:
        comun = ys[kmax].index
        log("    Cohorte comun (elegibles en k=%d): %d creditos" % (kmax, len(comun)))
        for k in [v for v in VENTANAS if v <= kmax]:
            # la etiqueta se mantiene la del punto de observacion de kmax para
            # que el objetivo sea identico en todas las ventanas
            y = ys[kmax]
            ds = armar(df, exo, comun, y, k, "completo")
            correr("corr_comun_completo", ds, df, k, mods, res, fold, oof,
                   {"D": D, "H": H, "k_objetivo": kmax})
    guardar(nombre, res, fold)
    if oof:
        o = pd.concat(oof, ignore_index=True)
        o.to_csv(os.path.join(SALIDA, "privado_oof_%s.csv" % nombre), index=False)
        por_plazo(o, nombre)


# =============================================================================
#  BLOQUE 3 - Objetivo original solo en creditos ya vencidos al corte
# =============================================================================
def bloque3(df, obs_max, exo, target, corte):
    base.titulo("BLOQUE 3 - Objetivo original, solo creditos ya vencidos")
    ult = df.groupby("c_simulacion")["d_vencimiento"].max()
    venc = ult[ult <= corte].index
    y = target["mora"].loc[target.index.intersection(venc)]
    res, fold, oof = [], [], []
    for k in VENTANAS:
        ids = obs_max.loc[obs_max.index.intersection(y.index)]
        ids = ids[ids >= k].index
        yk = y.loc[ids]
        if len(yk) < 100 or min(yk.sum(), (1 - yk).sum()) < 10:
            log("    k=%d: muestra insuficiente (n=%d, mora=%d)" % (k, len(yk), yk.sum()))
            continue
        for conjunto in ("completo", "sin_anticipo"):
            ds = armar(df, exo, ids, yk, k, conjunto)
            correr("vencidos_" + conjunto, ds, df, k, catalogo(), res, fold, oof)
    guardar("b3_original_solo_vencidos", res, fold)
    if oof:
        o = pd.concat(oof, ignore_index=True)
        o.to_csv(os.path.join(SALIDA, "privado_oof_b3.csv"), index=False)
        por_plazo(o, "b3_original_solo_vencidos")


def main():
    t0 = time.time()
    log("Corrida de revision | reps=%d | redes=%s | bloques=%s"
        % (REPS, REDES, BLOQUES))
    df, obs_max, exo, target, corte = preparar()
    if 0 in BLOQUES:
        bloque0(df, obs_max, exo, target, corte)
    if 1 in BLOQUES:
        bloque1(df, obs_max, exo, target, corte)
    if 2 in BLOQUES:
        # horizonte 12 meses (estandar); si la cartera es joven casi no habra
        # elegibles y se salta solo. Horizonte 6 meses como version principal
        # alternativa.
        bloque2(df, obs_max, exo, target, corte, 90, 365, "b2_corregido_90d_12m")
        bloque2(df, obs_max, exo, target, corte, 90, 180, "b2_corregido_90d_6m")
    if 3 in BLOQUES:
        bloque3(df, obs_max, exo, target, corte)
    if 4 in BLOQUES:
        sens = catalogo(con_redes=False, solo=["Reg. Logistica", "XGBoost (spw)", "Random Forest"])
        for D, H in ((60, 180),):
            bloque2(df, obs_max, exo, target, corte, D, H,
                    "b4_sensibilidad_%dd_%dm" % (D, H // 30), sens, ("completo",))
    if 5 in BLOQUES:
        # el desenlace "pagado" de la cartera es casi siempre un prepago: se
        # modela explicitamente como evento (riesgo competitivo)
        bloque2(df, obs_max, exo, target, corte, 90, 180, "b5_prepago_6m",
                conjuntos=("completo",), evento="prepago")
    if TESTS:
        pd.DataFrame(TESTS).to_csv(os.path.join(SALIDA, "tests_nadeau_bengio.csv"),
                                   index=False)
    log("\nListo en %.1f min. Enviar todo lo de %s EXCEPTO los archivos privado_*."
        % ((time.time() - t0) / 60, SALIDA))


if __name__ == "__main__":
    main()
