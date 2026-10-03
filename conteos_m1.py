# -*- coding: utf-8 -*-
"""conteos_m1.py - dos conteos descriptivos para la revision (no entrena modelos).
Uso (misma carpeta que analisis_mora_temprana.py y corrida_revision.py):
    DATOS_CSV=/ruta/base_real.csv python conteos_m1.py
Imprime y guarda resultados_revision/conteos_m1.txt (solo agregados)."""
import os
import numpy as np, pandas as pd
import analisis_mora_temprana as base
import corrida_revision as rev

df, obs_max, exo, target, corte = rev.preparar()
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# 1) Eventos a k = 3 (90+ dpd en 6 meses): ¿cuantos ya tenian una cuota impaga en t_3?
for k in (1, 2, 3, 5):
    y = rev.objetivo_corregido(df, obs_max, corte, k, 90, 180, target, "default")
    fw = base.features_ventana(df, k=k).reindex(y.index)
    imp = fw["w_n_impagas"] > 0
    p(f"k={k}: n={len(y)} eventos={int(y.sum())} | eventos con >=1 cuota impaga en t_k: "
      f"{int((imp & (y==1)).sum())} ({100*(imp[y==1]).mean():.1f}%) | "
      f"no-eventos con cuota impaga: {int((imp & (y==0)).sum())} ({100*(imp[y==0]).mean():.1f}%) | "
      f"tasa de evento si impaga: {100*y[imp].mean() if imp.any() else float('nan'):.1f}% / si al dia: {100*y[~imp].mean():.1f}%")

# 2) Clase 'en mora': dias de atraso en T de la cuota impaga mas antigua
imp = df[(df["pagada"] == 0) & (df["d_vencimiento"] <= corte)]
oldest = imp.groupby("c_simulacion")["d_vencimiento"].min()
dpd = (corte - oldest).dt.days
dpd = dpd.reindex(target.index[target["mora"] == 1])
bins = [-1, 30, 60, 90, 180, 10**6]; labs = ["1-30", "31-60", "61-90", "91-180", ">180"]
tab = pd.cut(dpd, bins=bins, labels=labs).value_counts().reindex(labs)
p("\nClase en mora (n=%d): dpd en T de la cuota impaga mas antigua" % dpd.notna().sum())
for l, v in tab.items(): p(f"  {l:>7}: {v:5d} ({100*v/dpd.notna().sum():.1f}%)")
p("  mediana dpd: %.0f" % dpd.median())
# fechas de pago en las ultimas semanas (indicio de rezago de carga)
pag = df[df["pagada"] == 1]
for d in (7, 14, 30, 60):
    p(f"pagos registrados en los ultimos {d} dias antes de T: {int((pag['d_pago'] > corte - pd.Timedelta(days=d)).sum())}")
open(os.path.join(rev.SALIDA, "conteos_m1.txt"), "w").write("\n".join(out))
