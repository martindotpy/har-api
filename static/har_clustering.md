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
import os
import zipfile
from pathlib import Path
from typing import Final

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
```


```python
dataset_url: Final[str] = (
    "https://archive.ics.uci.edu/static/public/779/harth.zip"
)
```


```python
response = httpx.get(
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
    temp_df = pd.read_csv(file)
    print(
        f"Archivo: {file}, Tamaño: {temp_df.shape}"
    )  # Imprime el tamaño de cada archivo
    df_list.append(temp_df)

# Combina los DataFrames
df = pd.concat(df_list, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Verifica el contenido del DataFrame después de cargar los archivos
print("Contenido del DataFrame después de cargar los archivos:")
print(df.head())
print(df.info())
```

    Archivo: data\harth\S006.csv, Tamaño: (408709, 8)
    Archivo: data\harth\S008.csv, Tamaño: (418989, 8)
    Archivo: data\harth\S009.csv, Tamaño: (154464, 8)
    Archivo: data\harth\S010.csv, Tamaño: (351649, 8)
    Archivo: data\harth\S012.csv, Tamaño: (382414, 8)
    Archivo: data\harth\S013.csv, Tamaño: (369077, 8)
    Archivo: data\harth\S014.csv, Tamaño: (366487, 8)
    Archivo: data\harth\S015.csv, Tamaño: (418392, 9)
    Archivo: data\harth\S016.csv, Tamaño: (355418, 8)
    Archivo: data\harth\S017.csv, Tamaño: (366609, 8)
    Archivo: data\harth\S018.csv, Tamaño: (322271, 8)
    Archivo: data\harth\S019.csv, Tamaño: (297945, 8)
    Archivo: data\harth\S020.csv, Tamaño: (371496, 8)
    Archivo: data\harth\S021.csv, Tamaño: (302247, 9)
    Archivo: data\harth\S022.csv, Tamaño: (337602, 8)
    Archivo: data\harth\S023.csv, Tamaño: (137646, 9)
    Archivo: data\harth\S024.csv, Tamaño: (170534, 8)
    Archivo: data\harth\S025.csv, Tamaño: (231729, 8)
    Archivo: data\harth\S026.csv, Tamaño: (195172, 8)
    Archivo: data\harth\S027.csv, Tamaño: (158584, 8)
    Archivo: data\harth\S028.csv, Tamaño: (165178, 8)
    Archivo: data\harth\S029.csv, Tamaño: (178716, 8)
    Contenido del DataFrame después de cargar los archivos:
                    timestamp    back_x    back_y    back_z   thigh_x   thigh_y  \
    0 2019-01-12 00:00:00.000 -0.760242  0.299570  0.468570 -5.092732 -0.298644   
    1 2019-01-12 00:00:00.010 -0.530138  0.281880  0.319987  0.900547  0.286944   
    2 2019-01-12 00:00:00.020 -1.170922  0.186353 -0.167010 -0.035442 -0.078423   
    3 2019-01-12 00:00:00.030 -0.648772  0.016579 -0.054284 -1.554248 -0.950978   
    4 2019-01-12 00:00:00.040 -0.355071 -0.051831 -0.113419 -0.547471  0.140903   
    
        thigh_z  label  index  Unnamed: 0  
    0  0.709439      6    NaN         NaN  
    1  0.340309      6    NaN         NaN  
    2 -0.515212      6    NaN         NaN  
    3 -0.221140      6    NaN         NaN  
    4 -0.653782      6    NaN         NaN  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6461328 entries, 0 to 6461327
    Data columns (total 10 columns):
     #   Column      Dtype         
    ---  ------      -----         
     0   timestamp   datetime64[ns]
     1   back_x      float64       
     2   back_y      float64       
     3   back_z      float64       
     4   thigh_x     float64       
     5   thigh_y     float64       
     6   thigh_z     float64       
     7   label       int64         
     8   index       float64       
     9   Unnamed: 0  float64       
    dtypes: datetime64[ns](1), float64(8), int64(1)
    memory usage: 493.0 MB
    None



```python
print("Número de valores nulos en cada columna:")
print(df.isna().sum())
```

    Número de valores nulos en cada columna:
    timestamp           0
    back_x              0
    back_y              0
    back_z              0
    thigh_x             0
    thigh_y             0
    thigh_z             0
    label               0
    index         5740689
    Unnamed: 0    6323682
    dtype: int64



```python
df = df.drop(columns=["Unnamed: 0", "index"], errors="ignore")
print(
    f"Tamaño del DataFrame después de eliminar columnas no necesarias: {df.shape}"
)
```

    Tamaño del DataFrame después de eliminar columnas no necesarias: (6461328, 8)



```python
print(f"Tamaño antes de eliminar nulos: {df.shape}")
df = df.dropna()  # Eliminar filas con valores nulos
print(f"Tamaño después de eliminar nulos: {df.shape}")
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

print(f"Tamaño del DataFrame antes de la normalización: {df.shape}")
print("Contenido del DataFrame antes de la normalización:")
print(df.head())

# Verifica el contenido de las columnas específicas
print("Contenido de las columnas a normalizar:")
print(df[quantitative_cols].head())

# Normalización
if all(col in df.columns for col in quantitative_cols):
    if df[quantitative_cols].shape[0] > 0:
        scaler = StandardScaler()
        df[quantitative_cols] = scaler.fit_transform(df[quantitative_cols])
    else:
        print("Las columnas seleccionadas están vacías.")
else:
    print("Una o más columnas no existen en el DataFrame.")
```

    Tamaño del DataFrame antes de la normalización: (6461328, 8)
    Contenido del DataFrame antes de la normalización:
                    timestamp    back_x    back_y    back_z   thigh_x   thigh_y  \
    0 2019-01-12 00:00:00.000 -0.760242  0.299570  0.468570 -5.092732 -0.298644   
    1 2019-01-12 00:00:00.010 -0.530138  0.281880  0.319987  0.900547  0.286944   
    2 2019-01-12 00:00:00.020 -1.170922  0.186353 -0.167010 -0.035442 -0.078423   
    3 2019-01-12 00:00:00.030 -0.648772  0.016579 -0.054284 -1.554248 -0.950978   
    4 2019-01-12 00:00:00.040 -0.355071 -0.051831 -0.113419 -0.547471  0.140903   
    
        thigh_z  label  
    0  0.709439      6  
    1  0.340309      6  
    2 -0.515212      6  
    3 -0.221140      6  
    4 -0.653782      6  
    Contenido de las columnas a normalizar:
         back_x    back_y    back_z   thigh_x   thigh_y   thigh_z
    0 -0.760242  0.299570  0.468570 -5.092732 -0.298644  0.709439
    1 -0.530138  0.281880  0.319987  0.900547  0.286944  0.340309
    2 -1.170922  0.186353 -0.167010 -0.035442 -0.078423 -0.515212
    3 -0.648772  0.016579 -0.054284 -1.554248 -0.950978 -0.221140
    4 -0.355071 -0.051831 -0.113419 -0.547471  0.140903 -0.653782



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
outlier_rows_z = df[outliers_z]
print("Número de outliers detectados por Z-Score:", outlier_rows_z.shape[0])
print(outlier_rows_z)

# IQR (Rango Intercuartílico)
Q1 = df[quantitative_cols].quantile(0.25)
Q3 = df[quantitative_cols].quantile(0.75)
IQR = Q3 - Q1

# Define límites para detectar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detecta outliers
outliers_iqr = (
    (df[quantitative_cols] < lower_bound)
    | (df[quantitative_cols] > upper_bound)
).any(axis=1)

# Muestra las filas que son outliers
outlier_rows_iqr = df[outliers_iqr]
print("Número de outliers detectados por IQR:", outlier_rows_iqr.shape[0])
outlier_rows_iqr
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_14_0.png)
    


    Número de outliers detectados por Z-Score: 507343
                          timestamp    back_x    back_y    back_z   thigh_x  \
    0       2019-01-12 00:00:00.000  0.330293  1.353248  1.749055 -7.181078   
    16144   2019-01-12 00:02:43.100 -1.210939  3.050026  0.711515 -0.653571   
    18890   2019-01-12 00:03:13.800 -0.919255  3.363959  0.687889 -0.649150   
    20549   2019-01-12 00:03:35.410 -1.474371  4.412604  1.671156 -1.148571   
    23333   2019-01-12 00:04:03.250 -0.886602  0.163189  1.250953 -3.329005   
    ...                         ...       ...       ...       ...       ...   
    6459996 2019-01-12 00:59:30.400 -1.652779  0.660400 -0.663487 -3.570954   
    6459997 2019-01-12 00:59:30.420 -0.780554  1.381719 -1.068448 -3.534314   
    6460051 2019-01-12 00:59:31.500 -0.461791 -0.048244 -0.160798 -4.045322   
    6460052 2019-01-12 00:59:31.520 -0.430755  0.126012 -0.189580 -3.476626   
    6461242 2019-01-12 00:59:55.320 -0.787019 -0.103161 -0.511542 -1.197163   
    
              thigh_y   thigh_z  label  
    0       -0.822551  0.454455      6  
    16144    0.513943 -0.950917      1  
    18890    0.358385 -0.296835      1  
    20549    0.652607 -0.362975      1  
    23333    3.131830 -2.136228      1  
    ...           ...       ...    ...  
    6459996 -2.380443  0.808722      1  
    6459997  1.696624 -1.552426      1  
    6460051 -2.165495 -4.413067      1  
    6460052 -3.895751 -4.952361      1  
    6461242 -4.169148 -0.234375      3  
    
    [507343 rows x 8 columns]
    Número de outliers detectados por IQR: 1399305





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>back_x</th>
      <th>back_y</th>
      <th>back_z</th>
      <th>thigh_x</th>
      <th>thigh_y</th>
      <th>thigh_z</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-12 00:00:00.000</td>
      <td>0.330293</td>
      <td>1.353248</td>
      <td>1.749055</td>
      <td>-7.181078</td>
      <td>-0.822551</td>
      <td>0.454455</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-12 00:00:00.030</td>
      <td>0.625506</td>
      <td>0.129082</td>
      <td>0.315552</td>
      <td>-1.531676</td>
      <td>-2.501871</td>
      <td>-0.809751</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-12 00:00:00.040</td>
      <td>1.403332</td>
      <td>-0.166845</td>
      <td>0.153423</td>
      <td>0.075705</td>
      <td>0.308988</td>
      <td>-1.397501</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-01-12 00:00:00.050</td>
      <td>1.337504</td>
      <td>-0.359878</td>
      <td>0.398052</td>
      <td>-0.569856</td>
      <td>0.542315</td>
      <td>-1.102051</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2019-01-12 00:00:00.070</td>
      <td>-2.238628</td>
      <td>0.277729</td>
      <td>0.513598</td>
      <td>-1.314569</td>
      <td>-0.263245</td>
      <td>-0.586286</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6461243</th>
      <td>2019-01-12 00:59:55.340</td>
      <td>-0.754690</td>
      <td>0.538949</td>
      <td>-0.425195</td>
      <td>-1.462605</td>
      <td>-2.305023</td>
      <td>-1.132202</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6461244</th>
      <td>2019-01-12 00:59:55.360</td>
      <td>-0.383556</td>
      <td>0.968782</td>
      <td>-0.410470</td>
      <td>-1.005387</td>
      <td>1.599206</td>
      <td>-1.222416</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6461245</th>
      <td>2019-01-12 00:59:55.380</td>
      <td>-0.118463</td>
      <td>0.496703</td>
      <td>-0.279943</td>
      <td>-0.498276</td>
      <td>1.425740</td>
      <td>-0.844314</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6461247</th>
      <td>2019-01-12 00:59:55.420</td>
      <td>-0.361572</td>
      <td>-0.211942</td>
      <td>-0.245138</td>
      <td>-0.674850</td>
      <td>-1.987632</td>
      <td>-0.729889</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6461248</th>
      <td>2019-01-12 00:59:55.440</td>
      <td>-0.422352</td>
      <td>-0.068312</td>
      <td>-0.273920</td>
      <td>-0.641328</td>
      <td>-1.477920</td>
      <td>-0.417125</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>1399305 rows × 8 columns</p>
</div>




```python
display(
    Markdown("Primeras filas del DataFrame preprocesado:"),
    df.head(),
    Markdown("Resumen estadístico del DataFrame preprocesado:"),
    df.describe(),
    Markdown("Información del DataFrame preprocesado:"),
)
df.info()
```


Primeras filas del DataFrame preprocesado:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>back_x</th>
      <th>back_y</th>
      <th>back_z</th>
      <th>thigh_x</th>
      <th>thigh_y</th>
      <th>thigh_z</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-12 00:00:00.000</td>
      <td>0.330293</td>
      <td>1.353248</td>
      <td>1.749055</td>
      <td>-7.181078</td>
      <td>-0.822551</td>
      <td>0.454455</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-12 00:00:00.010</td>
      <td>0.939690</td>
      <td>1.276724</td>
      <td>1.341687</td>
      <td>2.387553</td>
      <td>0.684944</td>
      <td>-0.047014</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-12 00:00:00.020</td>
      <td>-0.757338</td>
      <td>0.863492</td>
      <td>0.006493</td>
      <td>0.893189</td>
      <td>-0.255631</td>
      <td>-1.209252</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-12 00:00:00.030</td>
      <td>0.625506</td>
      <td>0.129082</td>
      <td>0.315552</td>
      <td>-1.531676</td>
      <td>-2.501871</td>
      <td>-0.809751</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-01-12 00:00:00.040</td>
      <td>1.403332</td>
      <td>-0.166845</td>
      <td>0.153423</td>
      <td>0.075705</td>
      <td>0.308988</td>
      <td>-1.397501</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Resumen estadístico del DataFrame preprocesado:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>back_x</th>
      <th>back_y</th>
      <th>back_z</th>
      <th>thigh_x</th>
      <th>thigh_y</th>
      <th>thigh_z</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6461328</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
      <td>6.461328e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012-02-20 18:20:58.805005824</td>
      <td>-3.177651e-17</td>
      <td>1.977674e-17</td>
      <td>5.236261e-16</td>
      <td>-1.634220e-16</td>
      <td>-1.349886e-16</td>
      <td>-6.127270e-16</td>
      <td>6.783833e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2000-01-01 00:00:00</td>
      <td>-1.884322e+01</td>
      <td>-1.857654e+01</td>
      <td>-1.756076e+01</td>
      <td>-1.182271e+01</td>
      <td>-2.064145e+01</td>
      <td>-1.137744e+01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2000-01-01 01:23:47.680000</td>
      <td>-3.110112e-01</td>
      <td>-3.022347e-01</td>
      <td>-5.557191e-01</td>
      <td>-6.056117e-01</td>
      <td>-3.114007e-01</td>
      <td>-7.208682e-01</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2019-01-12 00:12:46.760000</td>
      <td>-2.382000e-01</td>
      <td>6.858545e-02</td>
      <td>8.753365e-02</td>
      <td>2.764562e-01</td>
      <td>3.025462e-02</td>
      <td>4.422277e-01</td>
      <td>7.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2019-01-12 00:49:20.500000</td>
      <td>1.924149e-01</td>
      <td>3.710298e-01</td>
      <td>5.917970e-01</td>
      <td>6.817516e-01</td>
      <td>3.451519e-01</td>
      <td>7.794594e-01</td>
      <td>7.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019-01-12 02:23:36.720000</td>
      <td>8.412968e+00</td>
      <td>2.814024e+01</td>
      <td>1.392467e+01</td>
      <td>1.372187e+01</td>
      <td>2.054025e+01</td>
      <td>1.091066e+01</td>
      <td>1.400000e+02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.143238e+01</td>
    </tr>
  </tbody>
</table>
</div>



Información del DataFrame preprocesado:


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6461328 entries, 0 to 6461327
    Data columns (total 8 columns):
     #   Column     Dtype         
    ---  ------     -----         
     0   timestamp  datetime64[ns]
     1   back_x     float64       
     2   back_y     float64       
     3   back_z     float64       
     4   thigh_x    float64       
     5   thigh_y    float64       
     6   thigh_z    float64       
     7   label      int64         
    dtypes: datetime64[ns](1), float64(6), int64(1)
    memory usage: 394.4 MB



```python
# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df[quantitative_cols])
df["PC1"] = principal_components[:, 0]
df["PC2"] = principal_components[:, 1]
```


```python
print(df.columns)
```

    Index(['timestamp', 'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y',
           'thigh_z', 'label', 'PC1', 'PC2'],
          dtype='object')


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
df[quantitative_cols].hist(figsize=(15, 10), bins=20)
plt.tight_layout()
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_22_0.png)
    



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


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_23_0.png)
    



```python
plt.figure(figsize=(8, 6))
sns.countplot(x="label", data=df)
plt.title("Distribución de Actividades")
plt.xlabel("Código de Actividad")
plt.ylabel("Cantidad de Muestras")
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_24_0.png)
    



```python
corr_matrix = df[quantitative_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de Correlación de Datos del Acelerómetro")
plt.show()
```


    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_25_0.png)
    



```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="label", data=df, palette="deep")
plt.title("PCA de Datos del Acelerómetro")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\IPython\core\pylabtools.py:170: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_26_1.png)
    


## **Aplicación de K-Means para Clustering**


El clustering es una técnica de aprendizaje no supervisado que se utiliza para
agrupar datos en función de sus características. En este caso, aplicaremos el
algoritmo K-Means para identificar patrones en los datos del acelerómetro. El
objetivo es agrupar las muestras en diferentes clústeres basados en las
características cuantitativas, lo que puede ayudarnos a entender mejor las
diferentes actividades representadas en el conjunto de datos.



```python
# Definir el número de clústeres
n_clusters = 4
```


```python
# Aplicar K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(df[quantitative_cols])
```


```python
# Visualizar los resultados del clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="cluster",
    data=df,
    palette="deep",
    style="label",
    markers=["o", "s", "D"],
)
plt.title("Clustering K-Means de Datos del Acelerómetro")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.show()
```

    C:\Users\alexr\AppData\Local\Temp\ipykernel_20128\4154459502.py:3: UserWarning: 
    The markers list has fewer values (3) than needed (12) and will cycle, which may produce an uninterpretable plot.
      sns.scatterplot(
    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\IPython\core\pylabtools.py:170: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](https://api-har.martindotpy.dev/api/notebook/har_clustering_files/har_clustering_31_1.png)
    



```python
# Mostrar el número de muestras en cada clúster
print("Número de muestras en cada clúster:")
print(df["cluster"].value_counts())
```

    Número de muestras en cada clúster:
    cluster
    1    2795979
    2    2536244
    0     994941
    3     134164
    Name: count, dtype: int64



```python
# Ruta a los archivos CSV (ajústalo a tu ruta local)
data_dir = "./data/harth"

# Cargar solo algunos archivos y pocas filas para evitar exceso de RAM
data_frames = []
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")][:3]
for file in files:
    df = pd.read_csv(os.path.join(data_dir, file), nrows=10000)
    data_frames.append(df)
data = pd.concat(data_frames, ignore_index=True)
```


```python
# Mapear etiquetas a nombres
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
data["activity"] = data["label"].map(label_mapping)
data.dropna(subset=["activity"], inplace=True)

# Seleccionar variables (puedes agregar más si deseas)
features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
X = data[features]
y = data["activity"]
```


```python
# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Definir el modelo MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42,
)
```


```python
# Entrenar el modelo
mlp.fit(X_train, y_train)
```

    c:\Users\alexr\.dev\har\api\.venv\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(





<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>MLPClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.neural_network.MLPClassifier.html">?<span>Documentation for MLPClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)</pre></div> </div></div></div></div>




```python
# Evaluar
y_pred = mlp.predict(X_test)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))
```

    Matriz de Confusión:
    [[1314    2   14    6   24]
     [   3  171    0   78   20]
     [  19    0   52    0   25]
     [   4   30    2 3783   41]
     [  43   33   22   33  281]]
    
    Informe de Clasificación:
                  precision    recall  f1-score   support
    
     cycling_sit       0.95      0.97      0.96      1360
       shuffling       0.72      0.63      0.67       272
       stairs_up       0.58      0.54      0.56        96
        standing       0.97      0.98      0.97      3860
         walking       0.72      0.68      0.70       412
    
        accuracy                           0.93      6000
       macro avg       0.79      0.76      0.77      6000
    weighted avg       0.93      0.93      0.93      6000
    


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

