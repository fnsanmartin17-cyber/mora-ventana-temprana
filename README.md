# Predicción temprana de mora en créditos automotrices

¿Cuánto se puede saber sobre un crédito mirando solo sus **primeras k cuotas**, en vez de
esperar a que madure? Este repositorio contiene el código y los resultados del estudio.

Felipe San Martín · Magíster en Data Science, Universidad San Sebastián (Chile)

---

## El repositorio son dos archivos

| Archivo | Qué es |
|---|---|
| **[`analisis_mora_temprana.py`](analisis_mora_temprana.py)** | Todo el análisis en un solo script, explicado paso a paso |
| **[`datos_sinteticos.csv`](datos_sinteticos.csv)** | La base con la que se puede ejecutar (79.141 filas) |

```bash
pip install pandas numpy scikit-learn xgboost
python analisis_mora_temprana.py
```

Versiones con las que se produjeron los resultados (Python 3.13): `pandas 2.3.3` ·
`numpy 2.3.5` · `scikit-learn 1.7.2` · `xgboost 3.2.0` · `torch 2.11.0` (opcional).

Escribe las tablas de los cuatro objetivos en `resultados/`. Tarda entre 15 y 40 minutos. Si
además tenés PyTorch instalado corre también la CNN y la LSTM; si no, las salta y avisa.

> ### ⚠️ Los datos del repositorio son sintéticos
>
> `datos_sinteticos.csv` **no contiene ninguna persona real**. La base original tiene el RUT
> de 2.154 clientes —dato personal bajo la Ley 19.628 y la Ley 21.719— e información
> comercial de la institución que la aportó, sujeta a acuerdo de confidencialidad. No puede
> publicarse, ni en un repositorio privado.
>
> La base sintética reproduce el esquema, las distribuciones y las cuatro mecánicas del
> dominio (ver abajo). **Correr el script sobre ella da resultados plausibles pero NO
> idénticos a los publicados en este README, que provienen de los datos reales.** No es una
> reproducción fallida: es lo esperado, porque los créditos no son los mismos.

---

## El problema en dos hechos

**1. El desbalance está invertido.** La mora es la clase *mayoritaria*, no la rara. Y la
población cambia con la ventana, porque un crédito otorgado hace poco no tiene todavía una
quinta cuota que mirar:

| Ámbito | Créditos | % mora |
|---|---|---|
| Techo (historial hasta la penúltima cuota) | 2.074 | 80,7 % |
| Ventana k = 1 | 2.188 | 81,7 % |
| Ventana k = 2 | 2.074 | 80,7 % |
| Ventana k = 3 | 1.972 | 79,7 % |
| Ventana k = 5 | 1.699 | 76,4 % |
| Ventana k = 10 | 925 | 56,8 % |

**2. Por eso el F1 absoluto engaña.** Un clasificador que responde «mora» siempre, sin mirar
un solo dato, obtiene sobre una proporción `p` de clase mayoritaria:

```
Accuracy = p     Recall = 1     Precision = p     F1 = 2p / (1 + p)
```

Con p = 81,7 % eso da **F1 = 0,899 sin predecir nada**. Cualquier F1 tiene que leerse contra
ese piso. Por eso todas las tablas de aquí abajo incluyen la fila del clasificador trivial.

---

## Resultados

> Las tablas de esta sección provienen de los **datos reales**.
> Validación: `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, agrupado por cliente.

### Objetivo 1 — El techo

Cuánto se puede saber usando todo el historial hasta la penúltima cuota. No es un modelo
útil (a esa altura el desenlace ya casi se conoce): es la referencia superior.

| Modelo | Accuracy | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Reg. Logística | 0,990 | 0,994 | 0,989 | 0,998 | 1,000 |
| Random Forest | 0,991 | 0,995 | 0,999 | 0,991 | 1,000 |
| XGBoost | 0,992 | 0,995 | 0,996 | 0,995 | 1,000 |
| SVM | 0,990 | 0,994 | 0,990 | 0,998 | 1,000 |
| CNN | 0,939 | 0,961 | 0,930 | 0,994 | 0,993 |
| LSTM | 0,816 | 0,877 | 0,827 | 0,939 | 0,895 |
| **Clasificador trivial** | 0,807 | 0,893 | 1,000 | 0,807 | — |

### Objetivo 2 — Seis modelos con una ventana de 5 cuotas

| Modelo | Accuracy | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Reg. Logística | 0,829 | 0,884 | 0,849 | 0,922 | 0,893 |
| Random Forest | 0,875 | **0,921** | 0,948 | 0,896 | 0,896 |
| XGBoost | 0,874 | 0,920 | 0,949 | 0,893 | 0,896 |
| SVM | 0,830 | 0,883 | 0,840 | 0,932 | 0,893 |
| CNN | 0,801 | 0,860 | 0,805 | 0,926 | 0,887 |
| LSTM | 0,742 | 0,810 | 0,721 | 0,924 | 0,847 |
| **Clasificador trivial** | 0,764 | 0,866 | 1,000 | 0,764 | — |

### Objetivo 3 — Qué pasa al variar la ventana (F1)

| Modelo | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| Reg. Logística | 0,793 | 0,824 | 0,838 | 0,884 | 0,930 |
| Random Forest | 0,898 | 0,904 | 0,907 | 0,921 | 0,934 |
| XGBoost | 0,898 | 0,903 | 0,905 | 0,920 | **0,939** |
| SVM | 0,802 | 0,801 | 0,852 | 0,883 | 0,934 |
| CNN | 0,748 | 0,805 | 0,827 | 0,847 | 0,889 |
| LSTM | 0,777 | 0,768 | 0,794 | 0,773 | 0,898 |
| **Clasificador trivial** | **0,899** | 0,893 | 0,887 | 0,866 | 0,724 |

Esta tabla es la más importante del trabajo, y hay que leerla con cuidado. En k = 1 **el
clasificador trivial gana**: su 0,899 iguala o supera a los seis modelos. No es que los
modelos sean malos ahí, es que el 81,7 % de mora regala ese número. Y a medida que la ventana
crece, el trivial **baja** (0,899 → 0,724) mientras los modelos suben. Se cruzan.

Mirando solo la fila de XGBoost, pasar de 1 a 10 cuotas parece valer apenas +0,041 de F1. Esa
lectura es engañosa, y de ahí sale el cuarto objetivo.

### Objetivo 4 — Cohorte común: los mismos 925 créditos en todas las ventanas

Los objetivos anteriores comparan ventanas sobre poblaciones distintas, así que mezclan el
efecto de la ventana con el del cambio de población. Aquí se fijan los mismos 925 créditos —
los que sobreviven a k = 10 — y se vuelve a medir. El piso trivial queda constante en 0,724,
porque la cohorte no cambia:

| Ventana | Créditos | % mora | Mejor modelo | F1 | Recall | Precision | Margen sobre el trivial |
|---|---|---|---|---|---|---|---|
| k = 1 | 925 | 56,8 % | XGBoost | 0,751 | 0,789 | 0,717 | +0,027 |
| k = 2 | 925 | 56,8 % | XGBoost | 0,795 | 0,829 | 0,765 | +0,071 |
| k = 3 | 925 | 56,8 % | XGBoost | 0,809 | 0,842 | 0,780 | +0,085 |
| k = 5 | 925 | 56,8 % | XGBoost | 0,837 | 0,867 | 0,810 | +0,113 |
| k = 10 | 925 | 56,8 % | XGBoost | 0,939 | 0,958 | 0,921 | **+0,215** |

**Sobre población fija, pasar de 1 a 10 cuotas observadas vale +0,188 de F1, no +0,041.** La
diferencia entre esas dos cifras es, en una línea, la conclusión del trabajo.

---

## Cómo está construido el análisis

### La ventana se mide en cuotas, no en días

`k = 5` significa «las primeras cinco cuotas del crédito», hayan tomado cinco meses o quince.

### El corte de observación, para no espiar el futuro

Cada crédito tiene su propio punto de observación: el vencimiento de la última cuota de su
ventana. Desde ahí, una cuota cuenta como pagada **solo si el pago ocurrió antes de ese
punto**; si el pago fue después, al momento de observar todavía no se sabría. Y una cuota que
sigue impaga no tiene atraso cero: tiene los días que lleva vencida a esa fecha.

Sin esta regla, un crédito que se pone al día en la cuota 20 contamina su propia ventana de
k = 5 con información del futuro.

### Las variables

**De comportamiento** (prefijo `w_`, once en total): tasa de pago, atraso máximo, medio y su
desviación, número y tasa de atrasos, racha máxima de atrasos, atraso de la primera cuota y
anticipo medio. **Exógenas** (doce): precio, pie, plazo, tasa, monto financiado, LTV, valor
de cuota estimado, marca, producto, tipo de crédito y seguro.

**Quedan fuera a propósito** las columnas agregadas que trae la fuente (`cuotas_pagadas`,
`max_dias_atraso`, `estado_pago_credito`): resumen todo el historial, incluido lo que pasa
después de la ventana, y filtran la respuesta.

### Las cuatro mecánicas del dominio

No son obvias, y un análisis que las ignore da resultados equivocados. El script las explica
en detalle; en resumen:

1. **La base es una foto en el tiempo.** Solo algunas cuotas habían vencido al extraerla. De
   ahí que la población caiga de 2.188 a 925 créditos, y que los que se caen sean
   justamente los que incumplieron temprano.
2. **El pago es un prefijo contiguo.** Un crédito paga las cuotas 1..L y se detiene; no hay
   cuotas pagadas salteadas.
3. **Una cuota impaga no tiene atraso.** Trae `dias_atraso = 0` y `d_pago = 2000-01-01`, un
   centinela. Leer esa columna sin filtrar por `pagada = 1` invierte el significado.
4. **El piso de −60 días es un prepago, no un comportamiento.** Al liquidar un crédito
   anticipadamente, todas las cuotas restantes se saldan en una sola fecha; como esa fecha
   queda congelada mientras los vencimientos avanzan, el atraso se hunde y se trunca en −60.
   En la cuota 1 el −60 no existe en ninguna de las dos clases y recién emerge desde la
   tercera. Esa es la razón de fondo por la que k = 1 rinde como el trivial y k = 5 despega.

---

## Licencia

Código bajo licencia MIT (ver [`LICENSE`](LICENSE)). Los resultados de este README se
comparten bajo CC BY 4.0. La base real del estudio no se distribuye.
