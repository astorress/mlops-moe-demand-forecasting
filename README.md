# 🔋 Mixture of Experts (MoE) para Pronóstico de Demanda Eléctrica del SIN Colombia

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](docker-compose.yml)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Sistema de pronóstico de demanda eléctrica horaria del SIN (Sistema Interconectado Nacional) de Colombia basado en Mixture of Experts (MoE).**

---

## 📑 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Pipeline de Notebooks](#pipeline-de-notebooks)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Documentación Detallada](#documentación-detallada)
- [Licencia](#-licencia)

---

## 🎯 Descripción General

### Problema

El **Sistema Interconectado Nacional (SIN)** de Colombia requiere pronósticos precisos de demanda eléctrica en horizontes de corto plazo (24h a 1 semana) para:

- Optimización del despacho de generación
- Planificación de mantenimiento
- Reducción de costos operacionales

La demanda presenta **múltiples patrones simultáneos**:

- **Diario (24h)**: ciclos intradía (madrugada baja, mediodía pico, noche media)
- **Semanal (168h)**: diferencia laboral vs. fines de semana
- **Estacional**: variaciones mensuales y anuales

### Solución: Mixture of Experts (MoE)

Un pronóstico que:

1. **Identifica dominios** mediante clustering (perfiles diarios × perfiles semanales)
2. **Especifica expertos** — un modelo optimizado por dominio
3. **asignación dinámica de expertos** — elige el experto más apropiado según el contexto actual

**Resultado:** Mayor precisión en escenarios complejos (festivos, comportamientos anómalos, cambios de patrón).

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS HORARIOS (52,608)                  │
│              API SINERGOX (XM S.A.) 2020-2025               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │  01: ADQUISICIÓN & EDA        │
         │  - Descarga API               │
         │  - Validación                 │
         │  - Análisis exploratorio      │
         └────────────────┬──────────────┘
                          ↓
         ┌───────────────────────────────┐
         │  02: PREPROCESAMIENTO         │
         │  - Outlier detection (STL)    │
         │  - Imputación adaptativa      │
         │  - 38 features engineer       │
         └────────────────┬──────────────┘
                          ↓
         ┌───────────────────────────────────────────────┐
         │  03: CLUSTERING                               │
         │  ├─ TYPE_DAY: perfiles 24h (K-means)          │
         │  └─ TYPE_WEEK: perfiles 168h (K-means)        │
         │     → Cartesian product: (type_day, type_week)│
         └────────────────┬──────────────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  04: ANÁLISIS & CLASIFICADORES       │
         │  - Validar clusters vs. calendario   │
         │  - Entrenar clasificadores           │
         │    (LogReg, RF, SVM, GradBoost)      │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  05: SELECCIÓN DE EXPERTOS           │
         │  - 5 familias × 5 expertos           │
         │  - Walk-forward validation           │
         │  - Homogéneo vs. Heterogéneo         │
         │  → 25 modelos finales                │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  06: DISEÑO DEL GATING               │
         │  - Hard gating (baseline)            │
         │  - Soft (distancia, probabilístico)  │
         │  - Adaptativo (confianza)            │
         │  → Mejor mecanismo seleccionado      │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  07: OPTIMIZACIÓN HIPERPARÁMETROS    │
         │  - Random Search + Successive Halving│
         │  - HPO por experto                   │
         │  - HPO del gating                    │
         └────────────────┬─────────────────────┘
                          ↓
         ┌──────────────────────────────────────┐
         │  08: ENSAMBLE PRODUCCIÓN             │
         │  - Reentrenamiento completo          │
         │  - Clase MoEForecaster               │
         │  - Serialización (.pkl)              │
         │  → Modelo listo para producción      │
         └────────────────┬─────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────┐
    │         EVALUACIÓN Y VALIDACIÓN                │
    ├────────────────────────────────────────────────┤
    │  9: Benchmark comparativo                      │
    │      - MoE vs. 30 modelos (3 familias)         │
    │      - Análisis (desempeño, eficiencia,        │
    │        interpretabilidad, robustez)            │
    └────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
mlops-moe-demand-forecasting/
├── README.md                          ← Este archivo
├── RESUMEN_NOTEBOOKS.md               ← Descripción detallada de cada notebook
├── Makefile                           ← Comandos: make up, make pipeline, make nb
├── docker-compose.yml / docker/       ← Entorno de ejecución (JupyterLab + runner)
├── requirements.txt                   ← Dependencias con versiones fijas
│
├── src/moe_forecaster/               ← Clase MoEForecaster (la importa el notebook 08)
│
├── data/                             ← Data utilizada
│   ├── clustering/
│   ├── processed/
│   └── raw/
│
├── models/                           ← Modelos entrenados
│   ├── classifiers/
│   ├── experts/
│   ├── gating/
│   ├── hyperparameters/
│   └── production/
│
├── results/                           ← Gráficos y reportes
│
└── notebooks/                         ← Pipeline Jupyter
    ├── 01_data_acquisition.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_clustering.ipynb
    ├── 04_cluster_analysis_classifiers.ipynb
    ├── 05_expert_selection.ipynb
    ├── 06_gating_design.ipynb
    ├── 07_hpo.ipynb
    ├── 08_production_assembly.ipynb
    └── 09_benchmark_comparison.ipynb
```

---

## 🔄 Pipeline de Notebooks

Cada notebook sigue la estructura estandarizada:

```
# Notebook NN -- Nombre

---

## Objetivo
[Qué hace el notebook]

## Entrada
[Datos/archivos que consume]

## Salida
[Archivos que genera]

---

## Instalación
## Configuración
## Imports
## Carga de Datos
## [Análisis Principal]
## Guardado de Archivos
## Resumen
```

### Flujo Secuencial

| NB     | Nombre              | Objetivo                          | Entrada                | Salida                 |
| ------ | ------------------- | --------------------------------- | ---------------------- | ---------------------- |
| **01** | Adquisición         | Descargar 5 años del SIN          | API SINERGOX           | `demand.parquet`       |
| **02** | Preprocesamiento    | Outliers, imputación, 38 features | `demand.parquet`       | `features.parquet`     |
| **03** | Clustering Dual     | TYPE_DAY + TYPE_WEEK              | `features.parquet`     | Asignaciones cluster   |
| **04** | Análisis Clusters   | Validar clusters vs. calendario   | features + clusters    | Clasificadores         |
| **05** | Selección Expertos  | 25 modelos (5×5)                  | features + classifiers | Modelos expertos       |
| **06** | Gating Design       | Hard/Soft/Adaptativo              | features + expertos    | Mejor gating           |
| **07** | HPO                 | Optimizar hiperparámetros         | features + modelos     | Parámetros optimizados |
| **08** | Ensamble Producción | MoEForecaster final               | features + HPO params  | `.pkl` + API           |
| **09** | Benchmark           | MoE vs. 30 modelos                | features + MoE         | Ranking exhaustivo     |

---

## 📦 Requisitos e Instalación

Todo corre en Docker con versiones fijas: no hace falta instalar Python ni dependencias en tu máquina.

| Requisito | Versión |
| --------- | ------- |
| Docker Desktop (o Docker Engine) con Compose v2 | ≥ 24 |
| Make | cualquiera |
| Memoria asignada a Docker | 8 GB mínimo |
| Conexión a internet | Solo para el notebook 01 (API de XM) |

```bash
git clone https://github.com/astorress/mlops-moe-demand-forecasting.git
cd mlops-moe-demand-forecasting
make build      # construye la imagen (Python 3.12 + requirements.txt), ~5 min la primera vez
```

La imagen usa PyTorch en CPU. Si tienes GPU NVIDIA, los notebooks usan CUDA automáticamente cuando está disponible (por ejemplo, en Google Colab).

---

## 🚀 Cómo Ejecutar

Los notebooks se ejecutan **en orden**: cada uno lee los archivos que generó el anterior (`data/`, `models/`, `results/`). Si falta un archivo, el notebook se detiene con un error que indica cuál ejecutar primero.

### Opción 1: Explorar con JupyterLab

```bash
make up         # JupyterLab en http://localhost:8888
make down       # detenerlo
```

Abre `notebooks/` y ejecuta los notebooks del 01 al 09.

### Opción 2: Ejecutar el pipeline sin abrir Jupyter

```bash
make pipeline                        # notebooks 01 a 08: datos → modelo de producción
make benchmark                       # notebook 09: MoE vs. modelos de referencia
make nb NB=03_clustering             # un solo notebook
```

Las copias ejecutadas (con sus salidas y gráficos) quedan en `outputs/notebooks/`.

### Tiempos aproximados (CPU, Docker en un Mac de 14 núcleos)

| Notebook | Tiempo |
| -------- | ------ |
| 01 Adquisición | ~45 min la primera vez (2.192 peticiones a la API de XM); después usa `data/raw/demand.parquet` |
| 02 Preprocesamiento | ~2 min |
| 03 Clustering | ~1,5 h (K-Means con DTW) |
| 04 Clasificadores | ~1,5 h (validación cruzada de 10 folds) |
| 05 a 09 | Varias horas; el 05 y el 09 entrenan redes LSTM/GRU, más rápidas con GPU |

Para volver a descargar los datos de XM: `FORCE_DOWNLOAD=1` en el entorno del notebook 01. Para borrar todo lo generado: `make clean`.

### Opción 3: Google Colab (con GPU)

```python
!git clone https://github.com/astorress/mlops-moe-demand-forecasting.git
%cd mlops-moe-demand-forecasting/notebooks
```

Luego abre cada notebook desde `notebooks/` y ejecútalo; la celda de instalación detecta Colab e instala `requirements.txt`. Habilita la GPU en *Entorno de ejecución → Cambiar tipo de entorno*.

### Después de la Ejecución

- **data/raw/** — Demanda horaria descargada de la API de XM (52.608 registros, 2020-2025)
- **data/processed/** — Matriz de variables (52.608 × 39: demanda + 38 variables)
- **data/clustering/** — Asignaciones de tipo de día y tipo de semana
- **models/** — Clasificadores, expertos, gating y el modelo de producción `models/production/moe_forecaster.pkl`
- **results/** — Gráficos y reportes en CSV/JSON

### Usar el modelo de producción

```python
import pickle, sys
sys.path.insert(0, "src")                 # en Docker ya está en el PYTHONPATH
from moe_forecaster import MoEForecaster  # necesario para deserializar el modelo

with open("models/production/moe_forecaster.pkl", "rb") as f:
    model = pickle.load(f)

preds, gating = model.predict_with_metadata(X)   # X: DataFrame con model.feature_cols
```

---

## 📖 Documentación Detallada

Para más información, consulta:

- **[RESUMEN_NOTEBOOKS.md](RESUMEN_NOTEBOOKS.md)** — Descripción completa de cada notebook y su contenido
- **Notebooks individuales** — Código fuente y comentarios detallados

---

## 👨‍💼 Autor

**Sebastián Torres**

- Email: [sebastiantorres8800@gmail.com](mailto:sebastiantorres8800@gmail.com)
- GitHub: [@astorress](https://github.com/astorress)

---

---

## 📄 Licencia

[MIT](LICENSE)
