# **Hard Clustering: Análisis de Patrones de Movimiento con K-Means 📊**

<span style="font-size: 1.5rem;">

🎓 **Universidad:** Fundación Universitaria Konrad Lorenz

</span>

<span style="font-size: 1.25rem;">

🏫 **Facultad:** Facultad de Matemáticas e Ingenierías

📖 **Curso:** Introducción a Big Data

</span>

<span style="font-size: 1.5rem;">

**🧑‍💻 Integrantes:**

</span>

- Ángel Stiven Pinzón Sánchez - 506221100
- Andrea Valentina Cubillos Pinto - 506231711
- Martín Alexander Ramos Yampufe - 506251051


## **Introducción**


**El Reconocimiento de Actividad Humana (HAR)** es una técnica ampliamente
utilizada en el monitoreo de la salud, el análisis del rendimiento deportivo y
el seguimiento de la actividad física mediante dispositivos portátiles. Estos
sistemas suelen basarse en datos obtenidos de acelerómetros y giroscopios, los
cuales permiten detectar patrones de movimiento y clasificar actividades.

Uno de los principales desafíos en HAR es la detección de inactividad prolongada
en condiciones de vida libre, un aspecto clave para la prevención de
enfermedades asociadas al sedentarismo. Sin embargo, la mayoría de los estudios
actuales dependen de modelos supervisados, los cuales requieren datos
etiquetados, lo que puede ser un proceso costoso y propenso a errores.

Para abordar esta limitación, en este proyecto aplicaremos K-Means, un algoritmo
de aprendizaje no supervisado, para analizar datos de acelerómetros y detectar
segmentos de tiempo con baja actividad física. Este enfoque nos permitirá
identificar patrones de inactividad sin necesidad de etiquetas previas,
facilitando su aplicación en el monitoreo de la salud y la detección temprana de
conductas sedentarias.


## **Objetivos**


El objetivo de este análisis es identificar patrones de movimiento a partir de
datos de acelerómetros, explorando relaciones entre variables y reduciendo la
dimensionalidad de los datos para facilitar su agrupamiento mediante
**K-Means**. Esto permitirá detectar segmentos de baja actividad física y
evaluar su utilidad en el Reconocimiento de Actividad Humana (HAR) para prevenir
enfermedades relacionadas con el sedentarismo.


## **Preprocesamiento de datos**



```python
import io
import math
import zipfile
from inspect import cleandoc
from pathlib import Path
from typing import Final

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import seaborn as sns
from IPython.display import Markdown
from scipy import stats
from scipy.stats import zscore
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
```


```python
dataset_url: Final[str] = (
    "https://drive.usercontent.google.com/download?id=1pawtHobYPmvHLKKJfbg12fqkMdG5rlkL&export=download&authuser=0&confirm=t&uuid=c5110138-278c-4dc3-9be0-a11aeaefd54d&at=ALoNOgn9aAhEQRUpJf90DRNzLwiP%3A1748493341099"
)
```


```python
response = requests.get(
    dataset_url,
    timeout=10,
)

# Delete the data directory if it already exists
data_dir = Path("./data")


def remove_file_or_directory(file_or_directory: Path) -> None:
    """Elimina un archivo o carpeta de forma recursiva."""
    if file_or_directory.is_dir():
        for file in file_or_directory.iterdir():
            remove_file_or_directory(file)
        file_or_directory.rmdir()
    else:
        file_or_directory.unlink()


if data_dir.exists():
    remove_file_or_directory(data_dir)


with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
    zip_file.extractall("./data")
```


```python
path = Path("./data/harth")
all_files = path.glob("S0*.csv")

df_list = []
for file in all_files:
    temp_df = pl.read_csv(file)

    # Eliminamos la columna 'Unnamed: 0' si existe y index
    if "index" in temp_df.columns:
        temp_df = temp_df.drop("index")
    if "" in temp_df.columns:
        temp_df = temp_df.drop("")

    display(
        Markdown(f"Archivo: {file}, Tamaño: {temp_df.shape}")
    )  # Imprime el tamaño de cada archivo
    df_list.append(temp_df)

display(
    Markdown(
        f"Número total de filas en todos los archivos: {sum(df.shape[0] for df in df_list)}"
    )
)

# Combina los DataFrames
df: pl.DataFrame = pl.concat(
    df_list,
)

# Convierte la columna timestamp a tipo datetime
df = df.with_columns(
    pl.col("timestamp").str.strptime(
        pl.Datetime, "%Y-%m-%d %H:%M:%S%.f", strict=False
    )
)

# Verifica el contenido del DataFrame después de cargar los archivos
display(
    Markdown("Contenido del DataFrame después de cargar los archivos:"),
    df.head(),
    df.describe(),
    df.schema.to_frame(),
    Markdown(f"{df.estimated_size() / (1024 * 1024):.2f} MB"),
)
```


Archivo: data\harth\S006.csv, Tamaño: (408709, 8)



Archivo: data\harth\S008.csv, Tamaño: (418989, 8)



Archivo: data\harth\S009.csv, Tamaño: (154464, 8)



Archivo: data\harth\S010.csv, Tamaño: (351649, 8)



Archivo: data\harth\S012.csv, Tamaño: (382414, 8)



Archivo: data\harth\S013.csv, Tamaño: (369077, 8)



Archivo: data\harth\S014.csv, Tamaño: (366487, 8)



Archivo: data\harth\S015.csv, Tamaño: (418392, 8)



Archivo: data\harth\S016.csv, Tamaño: (355418, 8)



Archivo: data\harth\S017.csv, Tamaño: (366609, 8)



Archivo: data\harth\S018.csv, Tamaño: (322271, 8)



Archivo: data\harth\S019.csv, Tamaño: (297945, 8)



Archivo: data\harth\S020.csv, Tamaño: (371496, 8)



Archivo: data\harth\S021.csv, Tamaño: (302247, 8)



Archivo: data\harth\S022.csv, Tamaño: (337602, 8)



Archivo: data\harth\S023.csv, Tamaño: (137646, 8)



Archivo: data\harth\S024.csv, Tamaño: (170534, 8)



Archivo: data\harth\S025.csv, Tamaño: (231729, 8)



Archivo: data\harth\S026.csv, Tamaño: (195172, 8)



Archivo: data\harth\S027.csv, Tamaño: (158584, 8)



Archivo: data\harth\S028.csv, Tamaño: (165178, 8)



Archivo: data\harth\S029.csv, Tamaño: (178716, 8)



Número total de filas en todos los archivos: 6461328



Contenido del DataFrame después de cargar los archivos:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>2019-01-12 00:00:00</td><td>-0.760242</td><td>0.29957</td><td>0.46857</td><td>-5.092732</td><td>-0.298644</td><td>0.709439</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.010</td><td>-0.530138</td><td>0.28188</td><td>0.319987</td><td>0.900547</td><td>0.286944</td><td>0.340309</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.020</td><td>-1.170922</td><td>0.186353</td><td>-0.16701</td><td>-0.035442</td><td>-0.078423</td><td>-0.515212</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.030</td><td>-0.648772</td><td>0.016579</td><td>-0.054284</td><td>-1.554248</td><td>-0.950978</td><td>-0.22114</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.040</td><td>-0.355071</td><td>-0.051831</td><td>-0.113419</td><td>-0.547471</td><td>0.140903</td><td>-0.653782</td><td>6</td></tr></tbody></table></div>



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 9)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>str</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>&quot;6461328&quot;</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td></tr><tr><td>&quot;null_count&quot;</td><td>&quot;0&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>&quot;2012-02-20 18:20:58.805005&quot;</td><td>-0.884957</td><td>-0.013261</td><td>-0.169378</td><td>-0.594888</td><td>0.020877</td><td>0.374916</td><td>6.783833</td></tr><tr><td>&quot;std&quot;</td><td>null</td><td>0.377592</td><td>0.231171</td><td>0.364738</td><td>0.626347</td><td>0.388451</td><td>0.736098</td><td>11.432381</td></tr><tr><td>&quot;min&quot;</td><td>&quot;2000-01-01 00:00:00&quot;</td><td>-8.0</td><td>-4.307617</td><td>-6.574463</td><td>-8.0</td><td>-7.997314</td><td>-8.0</td><td>1.0</td></tr><tr><td>&quot;25%&quot;</td><td>&quot;2000-01-01 01:23:47.680000&quot;</td><td>-1.002393</td><td>-0.083129</td><td>-0.37207</td><td>-0.974211</td><td>-0.100087</td><td>-0.155714</td><td>3.0</td></tr><tr><td>&quot;50%&quot;</td><td>&quot;2019-01-12 00:12:46.760000&quot;</td><td>-0.9749</td><td>0.002594</td><td>-0.137451</td><td>-0.421731</td><td>0.032629</td><td>0.700439</td><td>7.0</td></tr><tr><td>&quot;75%&quot;</td><td>&quot;2019-01-12 00:49:20.500000&quot;</td><td>-0.812303</td><td>0.07251</td><td>0.046473</td><td>-0.167876</td><td>0.154951</td><td>0.948675</td><td>7.0</td></tr><tr><td>&quot;max&quot;</td><td>&quot;2019-01-12 02:23:36.720000&quot;</td><td>2.291708</td><td>6.491943</td><td>4.909483</td><td>7.999756</td><td>7.999756</td><td>8.406235</td><td>140.0</td></tr></tbody></table></div>



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (0, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody></tbody></table></div>



394.37 MB



```python
display(Markdown("Número de valores nulos en cada columna:"), df.null_count())
```


Número de valores nulos en cada columna:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td><td>u32</td></tr></thead><tbody><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table></div>



```python
display(Markdown(f"Tamaño antes de eliminar nulos: {df.shape}"))

df = df.drop_nulls()  # Eliminar filas con valores nulos

Markdown(f"Tamaño después de eliminar nulos: {df.shape}")
```


Tamaño antes de eliminar nulos: (6461328, 8)





Tamaño después de eliminar nulos: (6461328, 8)




```python
quantitative_cols = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

display(
    Markdown(
        cleandoc(f"""
            Tamaño del DataFrame antes de la normalización: {df.shape}

            Contenido del DataFrame antes de la normalización:
        """),
    ),
    df.head(),
)

# Normalización
if all(col in df.columns for col in quantitative_cols):
    if df[quantitative_cols].shape[0] > 0:
        scaler = StandardScaler()
        df[quantitative_cols] = scaler.fit_transform(
            df[quantitative_cols].to_arrow()
        )
    else:
        display(Markdown("Las columnas seleccionadas están vacías."))
else:
    display(Markdown("Una o más columnas no existen en el DataFrame."))
```


Tamaño del DataFrame antes de la normalización: (6461328, 8)

Contenido del DataFrame antes de la normalización:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>2019-01-12 00:00:00</td><td>-0.760242</td><td>0.29957</td><td>0.46857</td><td>-5.092732</td><td>-0.298644</td><td>0.709439</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.010</td><td>-0.530138</td><td>0.28188</td><td>0.319987</td><td>0.900547</td><td>0.286944</td><td>0.340309</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.020</td><td>-1.170922</td><td>0.186353</td><td>-0.16701</td><td>-0.035442</td><td>-0.078423</td><td>-0.515212</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.030</td><td>-0.648772</td><td>0.016579</td><td>-0.054284</td><td>-1.554248</td><td>-0.950978</td><td>-0.22114</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.040</td><td>-0.355071</td><td>-0.051831</td><td>-0.113419</td><td>-0.547471</td><td>0.140903</td><td>-0.653782</td><td>6</td></tr></tbody></table></div>



```python
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))
for i, col in enumerate(quantitative_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()

# Z-Score
z_scores = stats.zscore(df[quantitative_cols])
abs_z_scores = abs(z_scores)
outliers_z = (abs_z_scores > 3).any(axis=1)  # noqa: PLR2004

# Muestra las filas que son outliers
outlier_rows_z = df.filter(outliers_z)
display(
    Markdown(
        f"Número de outliers detectados por Z-Score: ${outlier_rows_z.shape[0]}$",
    ),
    outlier_rows_z,
)

# IQR (Rango Intercuartílico)
Q1 = df[quantitative_cols].quantile(0.25)
Q3 = df[quantitative_cols].quantile(0.75)
IQR = Q3 - Q1

# Define límites para detectar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detecta outliers
outliers_iqr: list[bool] = (
    (df[quantitative_cols].to_numpy() < lower_bound.to_numpy())
    | (df[quantitative_cols].to_numpy() > upper_bound.to_numpy())
).any(axis=1)


# Muestra las filas que son outliers
outlier_rows_iqr = df.filter(outliers_iqr)
display(
    Markdown(
        f"Número de outliers detectados por IQR: ${outlier_rows_iqr.shape[0]}$"
    ),
    outlier_rows_iqr,
)
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_13_0.png)
    



Número de outliers detectados por Z-Score: $507343$



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (507_343, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>2019-01-12 00:00:00</td><td>0.330293</td><td>1.353248</td><td>1.749055</td><td>-7.181078</td><td>-0.822551</td><td>0.454455</td><td>6</td></tr><tr><td>2019-01-12 00:02:43.100</td><td>-1.210939</td><td>3.050026</td><td>0.711515</td><td>-0.653571</td><td>0.513943</td><td>-0.950917</td><td>1</td></tr><tr><td>2019-01-12 00:03:13.800</td><td>-0.919255</td><td>3.363959</td><td>0.687889</td><td>-0.64915</td><td>0.358385</td><td>-0.296835</td><td>1</td></tr><tr><td>2019-01-12 00:03:35.410</td><td>-1.474371</td><td>4.412604</td><td>1.671156</td><td>-1.148571</td><td>0.652607</td><td>-0.362975</td><td>1</td></tr><tr><td>2019-01-12 00:04:03.250</td><td>-0.886602</td><td>0.163189</td><td>1.250953</td><td>-3.329005</td><td>3.13183</td><td>-2.136228</td><td>1</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2019-01-12 00:59:30.400</td><td>-1.652779</td><td>0.6604</td><td>-0.663487</td><td>-3.570954</td><td>-2.380443</td><td>0.808722</td><td>1</td></tr><tr><td>2019-01-12 00:59:30.420</td><td>-0.780554</td><td>1.381719</td><td>-1.068448</td><td>-3.534314</td><td>1.696624</td><td>-1.552426</td><td>1</td></tr><tr><td>2019-01-12 00:59:31.500</td><td>-0.461791</td><td>-0.048244</td><td>-0.160798</td><td>-4.045322</td><td>-2.165495</td><td>-4.413067</td><td>1</td></tr><tr><td>2019-01-12 00:59:31.520</td><td>-0.430755</td><td>0.126012</td><td>-0.18958</td><td>-3.476626</td><td>-3.895751</td><td>-4.952361</td><td>1</td></tr><tr><td>2019-01-12 00:59:55.320</td><td>-0.787019</td><td>-0.103161</td><td>-0.511542</td><td>-1.197163</td><td>-4.169148</td><td>-0.234375</td><td>3</td></tr></tbody></table></div>



Número de outliers detectados por IQR: $1399305$



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1_399_305, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>2019-01-12 00:00:00</td><td>0.330293</td><td>1.353248</td><td>1.749055</td><td>-7.181078</td><td>-0.822551</td><td>0.454455</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.030</td><td>0.625506</td><td>0.129082</td><td>0.315552</td><td>-1.531676</td><td>-2.501871</td><td>-0.809751</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.040</td><td>1.403332</td><td>-0.166845</td><td>0.153423</td><td>0.075705</td><td>0.308988</td><td>-1.397501</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.050</td><td>1.337504</td><td>-0.359878</td><td>0.398052</td><td>-0.569856</td><td>0.542315</td><td>-1.102051</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.070</td><td>-2.238628</td><td>0.277729</td><td>0.513598</td><td>-1.314569</td><td>-0.263245</td><td>-0.586286</td><td>6</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2019-01-12 00:59:55.340</td><td>-0.75469</td><td>0.538949</td><td>-0.425195</td><td>-1.462605</td><td>-2.305023</td><td>-1.132202</td><td>3</td></tr><tr><td>2019-01-12 00:59:55.360</td><td>-0.383556</td><td>0.968782</td><td>-0.41047</td><td>-1.005387</td><td>1.599206</td><td>-1.222416</td><td>3</td></tr><tr><td>2019-01-12 00:59:55.380</td><td>-0.118463</td><td>0.496703</td><td>-0.279943</td><td>-0.498276</td><td>1.42574</td><td>-0.844314</td><td>3</td></tr><tr><td>2019-01-12 00:59:55.420</td><td>-0.361572</td><td>-0.211942</td><td>-0.245138</td><td>-0.67485</td><td>-1.987632</td><td>-0.729889</td><td>3</td></tr><tr><td>2019-01-12 00:59:55.440</td><td>-0.422352</td><td>-0.068312</td><td>-0.27392</td><td>-0.641328</td><td>-1.47792</td><td>-0.417125</td><td>3</td></tr></tbody></table></div>



```python
display(
    Markdown("Primeras filas del DataFrame preprocesado:"),
    df.head(),
    Markdown("Resumen estadístico del DataFrame preprocesado:"),
    df.describe(),
    Markdown("Información del DataFrame preprocesado:"),
    df.schema.to_frame(),
    Markdown(f"{df.estimated_size('mb')} MB"),
)
```


Primeras filas del DataFrame preprocesado:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>2019-01-12 00:00:00</td><td>0.330293</td><td>1.353248</td><td>1.749055</td><td>-7.181078</td><td>-0.822551</td><td>0.454455</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.010</td><td>0.93969</td><td>1.276724</td><td>1.341687</td><td>2.387553</td><td>0.684944</td><td>-0.047014</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.020</td><td>-0.757338</td><td>0.863492</td><td>0.006493</td><td>0.893189</td><td>-0.255631</td><td>-1.209252</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.030</td><td>0.625506</td><td>0.129082</td><td>0.315552</td><td>-1.531676</td><td>-2.501871</td><td>-0.809751</td><td>6</td></tr><tr><td>2019-01-12 00:00:00.040</td><td>1.403332</td><td>-0.166845</td><td>0.153423</td><td>0.075705</td><td>0.308988</td><td>-1.397501</td><td>6</td></tr></tbody></table></div>



Resumen estadístico del DataFrame preprocesado:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 9)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>str</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>&quot;6461328&quot;</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td><td>6.461328e6</td></tr><tr><td>&quot;null_count&quot;</td><td>&quot;0&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;mean&quot;</td><td>&quot;2012-02-20 18:20:58.805005&quot;</td><td>-2.0081e-13</td><td>1.8051e-16</td><td>3.2458e-14</td><td>-2.7717e-15</td><td>1.1915e-14</td><td>-1.0838e-13</td><td>6.783833</td></tr><tr><td>&quot;std&quot;</td><td>null</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>11.432381</td></tr><tr><td>&quot;min&quot;</td><td>&quot;2000-01-01 00:00:00&quot;</td><td>-18.843224</td><td>-18.576544</td><td>-17.560763</td><td>-11.822707</td><td>-20.641445</td><td>-11.377443</td><td>1.0</td></tr><tr><td>&quot;25%&quot;</td><td>&quot;2000-01-01 01:23:47.680000&quot;</td><td>-0.311011</td><td>-0.302234</td><td>-0.555719</td><td>-0.605612</td><td>-0.311401</td><td>-0.720868</td><td>3.0</td></tr><tr><td>&quot;50%&quot;</td><td>&quot;2019-01-12 00:12:46.760000&quot;</td><td>-0.2382</td><td>0.068585</td><td>0.087534</td><td>0.276456</td><td>0.030255</td><td>0.442228</td><td>7.0</td></tr><tr><td>&quot;75%&quot;</td><td>&quot;2019-01-12 00:49:20.500000&quot;</td><td>0.192415</td><td>0.37103</td><td>0.591797</td><td>0.681751</td><td>0.345152</td><td>0.779459</td><td>7.0</td></tr><tr><td>&quot;max&quot;</td><td>&quot;2019-01-12 02:23:36.720000&quot;</td><td>8.412968</td><td>28.140243</td><td>13.924667</td><td>13.721867</td><td>20.540245</td><td>10.910662</td><td>140.0</td></tr></tbody></table></div>



Información del DataFrame preprocesado:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (0, 8)</small><table border="1" class="dataframe"><thead><tr><th>timestamp</th><th>back_x</th><th>back_y</th><th>back_z</th><th>thigh_x</th><th>thigh_y</th><th>thigh_z</th><th>label</th></tr><tr><td>datetime[μs]</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody></tbody></table></div>



394.3681640625 MB



```python
# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[quantitative_cols].to_arrow())
df = df.with_columns(
    [
        pl.Series("PC1", principal_components[:, 0]),
        pl.Series("PC2", principal_components[:, 1]),
    ]
)

df.columns
```




    ['timestamp',
     'back_x',
     'back_y',
     'back_z',
     'thigh_x',
     'thigh_y',
     'thigh_z',
     'label',
     'PC1',
     'PC2']



## **Descripción del conjunto de datos**


El conjunto de datos contiene registros de acelerómetros con mediciones en
diferentes ejes para la espalda (back_x, back_y, back_z) y el muslo (thigh_x,
thigh_y, thigh_z), junto con una etiqueta (label) que clasifica la actividad.

- Se identificaron 6,461,328 registros en total.

- Se realizó un análisis exploratorio, mostrando la media cercana a 0 tras
  normalización, lo que sugiere datos estandarizados.

- Se detectaron 507,343 valores atípicos usando Z-Score, indicando posibles
  variaciones extremas en la actividad.

- Se aplicó **PCA**(Análisis de Componentes Principales ) para reducir la
  dimensionalidad a 2 componentes principales, facilitando la visualización de
  patrones en los datos.


## **Análisis Exploratorio de Datos (EDA)**


Se realizará un análisis exploratorio de los datos obtenidos por acelerómetros
para identificar patrones, anomalías y relaciones entre variables mediante
histogramas y matrices de correlación. Este proceso optimizará la selección de
características y la normalización de los datos para aplicar clustering con
K-Means de manera efectiva.



```python
plt.figure(figsize=(15, 10))
for i, col in enumerate(quantitative_cols):
    plt.subplot(2, 3, i + 1)
    plt.hist(df[col], bins=20)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_20_0.png)
    



```python
cols = 3
rows = math.ceil(len(quantitative_cols) / cols)
fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

axs = axs.flatten()

i = 0
for i, column in enumerate(quantitative_cols):
    sns.kdeplot(
        data=df,
        x=column,
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.5,
        ax=axs[i],
    )
    axs[i].set_title(f"Densidad de {column} según la actividad")

# Ocultar los gráficos vacíos
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_21_0.png)
    



```python
plt.figure(figsize=(8, 6))
sns.countplot(x="label", data=df)
plt.title("Distribución de Actividades")
plt.xlabel("Código de Actividad")
plt.ylabel("Cantidad de Muestras")
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_22_0.png)
    



```python
corr_matrix = df[quantitative_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de Correlación de Datos del Acelerómetro")
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_23_0.png)
    



```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="label", data=df, palette="deep")
plt.title("PCA de Datos del Acelerómetro")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(loc="upper right")
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_24_0.png)
    


## **Aplicación de K-Means para Clustering**


El clustering es una técnica de aprendizaje no supervisado que se utiliza para
agrupar datos en función de sus características. En este caso, aplicaremos el
algoritmo K-Means para identificar patrones en los datos del acelerómetro. El
objetivo es agrupar las muestras en diferentes clústeres basados en las
características cuantitativas, lo que puede ayudarnos a entender mejor las
diferentes actividades representadas en el conjunto de datos.



```python
# Mapear etiquetas a nombres usando Polars
label_mapping = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling_sit",
    14: "cycling_stand",
    130: "cycling_sit_inactive",
    140: "cycling_stand_inactive",
}

# Agregar columna de actividad usando Polars, asegurando que sea str
df = df.with_columns(
    pl.col("label")
    .map_elements(
        lambda x: label_mapping.get(x, "unknown"), return_dtype=pl.Utf8
    )
    .alias("activity")
)

# Reducimos la cantidad de filas a solo una centésima parte
size = df.shape[0] // 100
df = df.drop_nulls(subset=["activity"]).sample(n=size, seed=42)

# Seleccionar variables para el modelo
features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]


x = df[features].to_numpy()
y = df["activity"].to_numpy()
```


```python
# Normalización
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# ---- CLUSTERING CON MÉTODO DEL CODO ----
inertia = []
k_range = range(1, 6)  # ✅ menor rango, más rápido

for k in k_range:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

# Gráfica del Codo
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, "bo-")
plt.xlabel("Número de Clústeres (K)")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.grid(visible=True)
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_28_0.png)
    



```python
# Clustering final con K óptimo
optimal_k = 4
kmeans_final = MiniBatchKMeans(
    n_clusters=optimal_k, random_state=42, batch_size=1024
)
clusters = kmeans_final.fit_predict(x_scaled)

# Asignar cluster a la muestra, no al dataframe completo
df = df.with_columns([pl.Series("cluster", clusters)])

# Visualización PCA de los clusters
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title(f"Clusters KMeans con PCA ($K={optimal_k}$)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")
plt.grid(visible=True)
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_29_0.png)
    



```python
# Asegurarse de que y viene del mismo conjunto que X_scaled
y = df["activity"].to_numpy()

# ---- CLASIFICACIÓN CON MLP ----
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42,
)
mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)

display(
    Markdown("Matriz de Confusión:"),
    confusion_matrix(y_test, y_pred),
    Markdown("Informe de Clasificación:"),
)
print(classification_report(y_test, y_pred))
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(
    


Matriz de Confusión:



    array([[ 672,   29,   11,    2,    0,    3,    1,    9,    0,    2,    8,
              67],
           [  31,   29,    5,    0,    0,    0,    1,    6,    0,    0,    7,
               6],
           [  23,    6,   59,    3,    0,    3,    0,    0,    1,    1,    5,
              15],
           [   4,    1,    4,    2,    0,    0,    0,    1,    0,    0,    2,
               0],
           [   0,    0,    0,    0,  845,    2,    0,    2,    0,    0,    0,
               1],
           [   3,    0,    3,    0,    3,  468,    0,    6,    1,    1,    1,
              89],
           [  13,    1,    1,    0,    3,    1,   65,    3,    1,    0,  251,
             160],
           [  10,    1,    0,    0,    1,    0,    1, 5724,    0,    0,    5,
               7],
           [  11,    0,    2,    0,    0,    8,    0,    0,    5,    0,    5,
              99],
           [  24,    1,    3,    0,    0,    3,    1,    0,    1,    8,    8,
              94],
           [  12,    3,    3,    0,    0,    2,   24,   12,    0,    1, 1355,
              93],
           [  41,    8,   17,    2,    0,   51,   56,    5,   11,    3,  200,
            2059]])



Informe de Clasificación:


                            precision    recall  f1-score   support
    
               cycling_sit       0.80      0.84      0.82       804
      cycling_sit_inactive       0.37      0.34      0.35        85
             cycling_stand       0.55      0.51      0.53       116
    cycling_stand_inactive       0.22      0.14      0.17        14
                     lying       0.99      0.99      0.99       850
                   running       0.87      0.81      0.84       575
                 shuffling       0.44      0.13      0.20       499
                   sitting       0.99      1.00      0.99      5749
               stairs_down       0.25      0.04      0.07       130
                 stairs_up       0.50      0.06      0.10       143
                  standing       0.73      0.90      0.81      1505
                   walking       0.77      0.84      0.80      2453
    
                  accuracy                           0.87     12923
                 macro avg       0.62      0.55      0.56     12923
              weighted avg       0.86      0.87      0.86     12923
    
    


```python
# Revisar visualmente los outliers en las características
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features])
plt.title("Boxplot de características para detectar valores atípicos")
plt.xticks(rotation=45)
plt.grid(visible=True)
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_31_0.png)
    



```python
# Clasificación con datos actuales (con outliers)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)
mlp.fit(X_train_o, y_train_o)
y_pred_o = mlp.predict(X_test_o)
acc_original = accuracy_score(y_test_o, y_pred_o)

# Quitar outliers usando Z-score
z_scores = np.abs(zscore(df[features]))
filtered_entries = (z_scores < 3).all(axis=1)
data_no_outliers = df.filter(filtered_entries)

# Re-calcular datos
X_clean = scaler.fit_transform(data_no_outliers[features].to_numpy())
y_clean = data_no_outliers["activity"]
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)
mlp.fit(x_train_c, y_train_c)
y_pred_c = mlp.predict(x_test_c)
acc_clean = accuracy_score(y_test_c, y_pred_c)

# Comparar resultados
display(
    Markdown(
        cleandoc(f"""
            Accuracy con outliers: {acc_original:.4f}

            Accuracy sin outliers: {acc_clean:.4f}""")
    )
)

if acc_clean > acc_original:
    Markdown(
        cleandoc("""
            ✅ Los valores atípicos estaban afectando negativamente al modelo.

            👉 Es recomendable aplicar una estrategia para eliminarlos o mitigarlos.""")
    )

else:
    Markdown(
        "👍 Los valores atípicos no están afectando negativamente al modelo."
    )
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(
    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(
    


Accuracy con outliers: 0.8737

Accuracy sin outliers: 0.8680


1. Entrenamiento del modelo Nuevamente Al ya tener el modelo este puede aprender
   patrones en los datos:



```python
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(
    

2. Matriz de confusión: ¿Qué patrones acierta o falla el modelo?



```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot(cmap="viridis", xticks_rotation=45)
plt.title("Matriz de Confusión - MLP")
plt.grid(visible=False)
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_36_0.png)
    


Interpretación:

Pemite ver qué actividades físicas se predicen bien.

Si por ejemplo siempre confunde walking con jogging, hay un patrón de similitud
que puedes investigar más.


3. Análisis de importancia de características (permutación) Esto te dice qué
   variables (acelerómetro, giroscopio, etc.) son más importantes para predecir
   una actividad:



```python
result = permutation_importance(
    mlp, x_test, y_test, n_repeats=10, random_state=42
)
importances = result.importances_mean
feature_names = features  # Usa tu lista de nombres de columnas

# Graficar importancia de cada característica
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances)
plt.xlabel("Importancia Media")
plt.title("Importancia de características para predicción de actividad")
plt.grid(visible=True)
plt.tight_layout()
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_39_0.png)
    


4. Visualización PCA o t-SNE para detectar agrupamientos naturales Esto ayuda a
   ver si hay patrones de agrupamiento en los movimientos:



```python
x_tsne = TSNE(n_components=2, random_state=42).fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    x_tsne[:, 0],
    x_tsne[:, 1],
    c=df["activity"].cast(pl.Categorical).to_physical().to_numpy(),
    cmap="tab10",
    alpha=0.7,
)
plt.title("Visualización t-SNE de Actividades Físicas")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Actividad (códigos)")
plt.grid(visible=True)
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_41_0.png)
    


5. Conclusión: Relación con el problema planteado

Se identificaron patrones relevantes en los sensores del dispositivo (como la
aceleración en el eje X y el giroscopio en Z) que permiten predecir con
precisión actividades físicas como caminar, correr o estar sentado. El modelo
MLP alcanzó una precisión del X%, y se observaron agrupamientos claros entre
clases similares, lo que permite implementar una solución efectiva de
reconocimiento de actividad física en tiempo real.



```python
# Codificación de etiquetas si son categóricas
le = LabelEncoder()
y_encoded = le.fit_transform(y[: len(x_scaled)])

# División
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y_encoded, test_size=0.2, random_state=42
)
```


```python
# Modelo secuencial
model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(x_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(
            len(
                set(y_encoded)
            ),  # salida multiclase # type: ignore  # noqa: PGH003
            activation="softmax",
        ),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    


```python
# Entrenamiento
history = model.fit(
    x_train, y_train, epochs=25, batch_size=64, validation_split=0.2, verbose=1
)
```

    Epoch 1/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.6189 - loss: 1.2967 - val_accuracy: 0.8138 - val_loss: 0.6079
    Epoch 2/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.7988 - loss: 0.6496 - val_accuracy: 0.8356 - val_loss: 0.5367
    Epoch 3/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8191 - loss: 0.5875 - val_accuracy: 0.8432 - val_loss: 0.5049
    Epoch 4/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8296 - loss: 0.5530 - val_accuracy: 0.8491 - val_loss: 0.4894
    Epoch 5/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8358 - loss: 0.5346 - val_accuracy: 0.8543 - val_loss: 0.4768
    Epoch 6/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8408 - loss: 0.5122 - val_accuracy: 0.8554 - val_loss: 0.4697
    Epoch 7/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8420 - loss: 0.5014 - val_accuracy: 0.8570 - val_loss: 0.4611
    Epoch 8/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8426 - loss: 0.5024 - val_accuracy: 0.8574 - val_loss: 0.4587
    Epoch 9/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8418 - loss: 0.5034 - val_accuracy: 0.8579 - val_loss: 0.4525
    Epoch 10/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8477 - loss: 0.4829 - val_accuracy: 0.8606 - val_loss: 0.4513
    Epoch 11/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8523 - loss: 0.4750 - val_accuracy: 0.8570 - val_loss: 0.4553
    Epoch 12/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8511 - loss: 0.4743 - val_accuracy: 0.8629 - val_loss: 0.4446
    Epoch 13/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8476 - loss: 0.4711 - val_accuracy: 0.8640 - val_loss: 0.4410
    Epoch 14/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8528 - loss: 0.4649 - val_accuracy: 0.8626 - val_loss: 0.4374
    Epoch 15/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8534 - loss: 0.4655 - val_accuracy: 0.8656 - val_loss: 0.4331
    Epoch 16/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8538 - loss: 0.4592 - val_accuracy: 0.8633 - val_loss: 0.4377
    Epoch 17/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8531 - loss: 0.4641 - val_accuracy: 0.8642 - val_loss: 0.4317
    Epoch 18/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8545 - loss: 0.4599 - val_accuracy: 0.8641 - val_loss: 0.4323
    Epoch 19/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8557 - loss: 0.4574 - val_accuracy: 0.8631 - val_loss: 0.4332
    Epoch 20/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8587 - loss: 0.4492 - val_accuracy: 0.8643 - val_loss: 0.4326
    Epoch 21/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8583 - loss: 0.4477 - val_accuracy: 0.8657 - val_loss: 0.4242
    Epoch 22/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8581 - loss: 0.4537 - val_accuracy: 0.8675 - val_loss: 0.4261
    Epoch 23/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8556 - loss: 0.4545 - val_accuracy: 0.8652 - val_loss: 0.4262
    Epoch 24/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8570 - loss: 0.4476 - val_accuracy: 0.8633 - val_loss: 0.4288
    Epoch 25/25
    [1m647/647[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8552 - loss: 0.4455 - val_accuracy: 0.8653 - val_loss: 0.4279
    


```python
# Evaluación
y_pred = model.predict(x_test)
y_pred_labels = y_pred.argmax(axis=1)

display(
    Markdown("Matriz de Confusión:"),
    confusion_matrix(y_test, y_pred_labels),
    Markdown("Informe de Clasificación:"),
)
print(classification_report(y_test, y_pred_labels))
```

    [1m404/404[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step
    


Matriz de Confusión:



    array([[ 717,    0,    1,    0,    0,    2,    0,   14,    0,    0,    9,
              61],
           [  63,    0,    2,    0,    1,    0,    0,    6,    0,    0,   11,
               2],
           [  57,    0,   31,    0,    0,    1,    0,    0,    0,    0,    6,
              21],
           [  12,    0,    0,    0,    0,    0,    0,    1,    0,    0,    1,
               0],
           [   1,    0,    0,    0,  844,    0,    0,    2,    0,    0,    0,
               3],
           [   7,    0,    2,    0,    7,  427,    0,    9,    0,    0,    5,
             118],
           [  14,    0,    0,    0,    2,    1,    7,    3,    0,    0,  268,
             204],
           [  17,    0,    0,    0,    2,    0,    0, 5720,    0,    0,    3,
               7],
           [  16,    0,    1,    0,    0,   11,    0,    0,    0,    0,    6,
              96],
           [  42,    0,    3,    0,    0,    0,    0,    1,    0,    0,   11,
              86],
           [  22,    1,    0,    0,    0,    2,    4,   14,    0,    0, 1355,
             107],
           [  75,    0,   11,    0,    0,   29,    2,    7,    0,    0,  215,
            2114]])



Informe de Clasificación:


                  precision    recall  f1-score   support
    
               0       0.69      0.89      0.78       804
               1       0.00      0.00      0.00        85
               2       0.61      0.27      0.37       116
               3       0.00      0.00      0.00        14
               4       0.99      0.99      0.99       850
               5       0.90      0.74      0.81       575
               6       0.54      0.01      0.03       499
               7       0.99      0.99      0.99      5749
               8       0.00      0.00      0.00       130
               9       0.00      0.00      0.00       143
              10       0.72      0.90      0.80      1505
              11       0.75      0.86      0.80      2453
    
        accuracy                           0.87     12923
       macro avg       0.51      0.47      0.46     12923
    weighted avg       0.84      0.87      0.84     12923
    
    

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
    

Finalmente, compilamos el modelo para su uso dentro de la aplicación rest.



```python
build_path = Path("..", "build")

if build_path.exists():
    remove_file_or_directory(build_path)

build_path.mkdir(parents=True, exist_ok=True)

joblib.dump(mlp, build_path / "mlp_model.pkl")
None
```

## **Interpretación de resultados**


### **Interpretación de los Resultados del Clustering**

Se ha aplicado el algoritmo **K-Means** con **4 clústeres** sobre los datos para
identificar patrones dentro del conjunto de datos. A continuación, se
interpretan los resultados obtenidos:

#### **1. Aplicación del Clustering**

El modelo K-Means fue entrenado con los datos numéricos, asignando cada muestra
a uno de los cuatro clústeres definidos. Esto permitió segmentar el conjunto de
datos en grupos con características similares.

#### **2. Visualización de los Clústeres**

El gráfico generado representa los datos en función de dos componentes
principales (PC1 y PC2), facilitando su visualización en un espacio
bidimensional. Cada color representa un clúster diferente, lo que permite
observar cómo el algoritmo ha distribuido los datos.

#### **3. Distribución de las Muestras por Clúster**

Se observó que los tamaños de los clústeres varían significativamente:

- **Clúster 0:** 3,345,319 muestras.
- **Clúster 1:** 2,702,296 muestras.
- **Clúster 2:** 278,541 muestras.
- **Clúster 3:** 135,172 muestras.

La diferencia en el número de muestras por clúster sugiere que los datos no
están distribuidos uniformemente, lo que puede ser indicativo de estructuras o
patrones particulares en los datos.

#### **4. Utilidad del Clustering**

El clustering es una técnica útil para analizar datos sin etiquetas previas. En
este caso, su aplicación podría ayudar a:

- **Identificar patrones de movimiento** en los datos del acelerómetro.
- **Reducir la complejidad** del análisis al segmentar el conjunto de datos en
  grupos representativos.
- **Detectar anomalías**, ya que los clústeres más pequeños pueden representar
  eventos inusuales.
- **Facilitar la exploración de datos** sin necesidad de etiquetas predefinidas.

El uso de K-Means permitió obtener una segmentación efectiva de los datos,
proporcionando información valiosa para análisis posteriores.


## **Conclusiones y siguientes pasos**


### Conclusiones

- Se confirman correlaciones en las mediciones y se demuestra la utilidad de
  K-Means para segmentar datos. Se sugiere evaluar si 4 clústeres es óptimo,
  mejorar el preprocesamiento, probar modelos avanzados como Autoencoders o
  CNNs, y optimizar el procesamiento de grandes volúmenes de datos. Eficacia de
  K-Means

- K-Means logra segmentar los datos de aceleración en distintos grupos sin
  necesidad de etiquetas previas, demostrando su potencial para detectar
  patrones de inactividad en la vida diaria. Sin embargo, la selección del
  número de clústeres requiere mayor optimización para garantizar una
  segmentación más precisa. Impacto de la reducción de dimensionalidad

- El uso de PCA ayudó a mejorar la interpretación de los datos y facilitó la
  visualización de los clústeres, lo que sugiere que técnicas de reducción de
  dimensionalidad son clave en el preprocesamiento. Se podrían evaluar otras
  técnicas como t-SNE o UMAP para mejorar la representación de los datos.
  Diferencias en la correlación entre sensores

- Se identificaron correlaciones entre los sensores de la espalda y el muslo, lo
  que indica que la actividad del usuario afecta de manera diferente cada zona
  del cuerpo. Esto sugiere que futuros modelos podrían incorporar relaciones
  entre múltiples sensores para mejorar la detección de actividad. Limitaciones
  del modelo y mejoras futuras

- K-Means, al ser un método basado en distancia, puede no capturar completamente
  la variabilidad en los datos de acelerometría. Modelos más avanzados como
  redes neuronales recurrentes (RNN), Autoencoders o CNNs podrían ser más
  efectivos en la detección de patrones complejos. También se podría evaluar la
  combinación de K-Means con técnicas supervisadas para refinar la
  clasificación. Aplicaciones prácticas y futuras investigaciones

- La detección de inactividad con este enfoque puede aplicarse en monitoreo de
  salud, prevención de sedentarismo y estudios de ergonomía. Futuros estudios
  podrían analizar la relación entre los clústeres y eventos específicos de
  inactividad, para validar la utilidad del método en entornos reales.


### Siguientes pasos


- Verificar si $k = 4$ es el valor óptimo para la aplicación de K-Means,
  utilizando técnicas de evaluación como el método del codo o el coeficiente de
  silueta.
- Confirmar que los valores atípicos no afectan negativamente al modelo; de ser
  así, replantear la estrategia de procesamiento.
- Explorar el uso de redes neuronales para aprovechar de forma óptima la alta
  dimensionalidad de los datos.
- Dado el elevado número de registros en el dataset, el consumo de recursos y el
  tiempo de procesamiento pueden volverse ineficientes. Se recomienda considerar
  herramientas específicas para el manejo de grandes volúmenes de datos.
- Identificar patrones que permitan predecir los tipos de movimientos o la
  actividad física y con ello darle respuesta al problema planteado.

