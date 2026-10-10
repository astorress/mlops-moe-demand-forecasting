# 📊 Resumen de Notebooks — Proyecto MoE (Mixture of Experts)

## Predicción Horaria de Demanda Eléctrica del SIN Colombia

---

## 📋 Descripción por Notebook

### **01_data_acquisition.ipynb**

**Objetivo:** Descargar y validar 5 años de datos horarios de demanda eléctrica del SIN desde la API pública de SINERGOX (XM). Incluye análisis exploratorio completo: patrones temporales, estacionariedad y EDA.

**Entrada:** API SINERGOX (XM S.A.) — Demanda Real del SIN horaria (2020–2025)

**Salida:** `data/raw/demand.parquet` (52,608 registros) + `data/raw/metadata.json`

**Contenido Técnico:**

- Descarga con reintentos exponenciales y validación de completitud (24 registros/día)
- EDA: serie temporal, distribuciones por hora/día/mes, heatmap hora×día_semana
- ACF/PACF hasta 336 lags, descomposición STL (períodos 24 y 168)
- Pruebas estacionariedad (ADF, KPSS)

---

### **02_preprocessing.ipynb**

**Objetivo:** Transformar serie bruta en dataset enriquecido: detección de outliers (STL), imputación adaptativa, ingeniería de 38 variables (calendario, Fourier, rezagos, móviles). Garantizar causalidad (sin data leakage).

**Entrada:** `data/raw/demand.parquet`

**Salida:** `data/processed/features.parquet` (52,608 × 39) + `data/processed/feature_names.json`

**Contenido Técnico:**

1. **Outliers** — STL residual con 3×IQR, categorización (error_medición, falla_equipo, pico_real)
2. **Imputación adaptativa** — Lineal (≤3h), spline (4–12h), días similares (13–48h), múltiple (>48h)
3. **Variables (38)** — Calendario (sin/cos), festivos, Fourier (P=24,168), lags {1,2,3, 23,24,25, 47,48,49, 167,168,169}, móviles [24,168,336]h
4. **MSTL + ADF/KPSS** — Validación de estacionariedad

---

### **03_clustering.ipynb**

**Objetivo:** Definir dominios de expertos mediante clustering independiente en dos dimensiones: TYPE_DAY (perfiles diarios 24h) y TYPE_WEEK (perfiles semanales 168h).

**Entrada:** `data/processed/features.parquet`

**Salida:** Asignaciones de cluster por hora (type_day, type_week) + centroides de clusters

**Contenido Técnico:**

- KMeans en estadísticas horarias (TYPE_DAY) y diarias (TYPE_WEEK)
- Selección óptima por silhueta + inercia (k=2–10)
- Visualización: centroides, distribución temporal, heatmaps de demanda

---

### **04_cluster_analysis_classifiers.ipynb**

**Objetivo:** Validar que los clusters capturan patrones reales vs. calendario. Entrenar clasificadores multi-configuración para predecir cluster assignment desde features.

**Entrada:** `data/processed/features.parquet` + asignaciones de cluster

**Salida:** Modelos clasificadores entrenados + reportes de accuracy/ROC-AUC

**Contenido Técnico:**

- Contingencia: type_day vs. hora_día/día_semana; type_week vs. mes
- Clasificadores con 3 configuraciones: calendario only, autorregresivos only, todas las variables
- Métricas: accuracy, F1 macro, ROC-AUC
- Análisis de confianza y calibración

---

### **05_expert_selection.ipynb**

**Objetivo:** Seleccionar e identificar los mejores modelos para cada dominio (cluster). Evaluar 5 familias de modelos (5 expertos/familia) con walk-forward validation. Comparar enfoques homogéneo vs. heterogéneo.

**Entrada:** `data/processed/features.parquet` + asignaciones de cluster + clasificadores entrenados

**Salida:** Modelos expertos optimizados (5 familias × 5 expertos = 25 modelos) + tabla de desempeño por experto

**Contenido Técnico:**

- Motor walk-forward con ventanas Train/Validation/Test
- Evaluación homogénea: mismo modelo para todos los expertos (baseline)
- Evaluación heterogénea: mejor modelo por experto entre todas las familias
- Restricciones bias-varianza por cluster
- Métricas: MAE, RMSE, MAPE

---

### **06_gating_design.ipynb**

**Objetivo:** Diseñar y comparar 5 mecanismos de routing: hard, soft (distancia), soft (probabilístico), adaptativo, y seleccionar el mejor. Optimizar trade-off entre complejidad y rendimiento.

**Entrada:** features + asignaciones de cluster + clasificadores + modelos expertos

**Salida:** Mecanismo de routing (gating) seleccionado (hard/soft/adaptativo) + hiperparámetros optimizados

**Contenido Técnico:**

- **Hard gating**: asignación determinística al cluster más cercano
- **Soft (distancia)**: ponderación por proximidad a centroides
- **Soft (probabilístico)**: pesas = P(cluster | features)
- **Adaptativo**: confianza adicional del router con fallback
- Comparación integral y validación cruzada

---

### **07_hpo.ipynb**

**Objetivo:** Optimizar hiperparámetros de modelos expertos y gating mediante Random Search + Successive Halving. Generar curvas de convergencia y tabla de mejores parámetros por experto.

**Entrada:** features + modelos expertos + gating mechanism

**Salida:** Hiperparámetros optimizados (JSON) + modelos re-entrenados con mejores parámetros

**Contenido Técnico:**

- Random Search + Successive Halving (HalvingRandomSearchCV)
- Espacios de búsqueda específicos por familia (regularización, depth, learning_rate, etc.)
- HPO independiente por experto
- Curvas de convergencia (MAE vs. ronda halving)
- HPO del gating (umbral de confianza, temperatura softmax)

---

### **08_production_assembly.ipynb**

**Objetivo:** Ensamblar sistema MoE final: reentrenamiento con todos los datos, clase wrapper MoEForecaster, prueba de sanidad sobre las últimas 500 horas y serialización.

**Entrada:** features + modelos optimizados + gating + hiperparámetros

**Salida:** Modelo MoE serializado (`models/production/moe_forecaster.pkl`) + model card y configuración en JSON

**Contenido Técnico:**

- Reentrenamiento con todos los datos disponibles
- Clase MoEForecaster (`src/moe_forecaster/forecaster.py`): métodos `predict()`, `predict_with_metadata()` (expertos y modo de gating por hora) y `get_model_card()`
- Pronóstico 168h (1 semana) end-to-end
- Visualización final: serie histórica + pronóstico + barras de expertos asignados

---

### **09_benchmark_comparison.ipynb**

**Objetivo:** Comparar 30 modelos (3 familias × 10) vs. MoE. Análisis según: desempeño predictivo, eficiencia computacional, interpretabilidad (SHAP + routing), robustez (stress test COVID).

**Entrada:** features + modelo MoE

**Salida:** Tabla maestra + ranking por métrica + análisis de interpretabilidad + reporte de robustez

**Contenido Técnico:**

- **30 Modelos**: Estadística, ML, DL
- Random Search HPO por modelo, walk-forward validation
- **Análisis**:
  1. Desempeño: MAE, RMSE, MAPE
  2. Eficiencia: tiempo entrenamiento + predicción
  3. Interpretabilidad: escala 1–5, SHAP values
  4. Robustez: stress test COVID-19, períodos críticos, degradación
- Exportación de resultados

---

## 🔍 Análisis de Estructura

### **Estructura Estandarizada**

Todos los notebooks ahora siguen este patrón inicial:

```
┌─────────────────────────────────────────────┐
│ # Notebook NN -- Nombre                     │
│ ---                                         │
│ ## Objetivo                                 │
│ ## Entrada                                  │
│ ## Salida                                   │
├─────────────────────────────────────────────┤
│ ## Instalación                              │
│ ## Configuración                            │
│ ## Imports                                  │
│ ## Carga de Datos                           │
├─────────────────────────────────────────────┤
│ ## [Análisis Principal]                     │
├─────────────────────────────────────────────┤
│ ## Guardado de Archivos                     │
│ ## Resumen                                  │
└─────────────────────────────────────────────┘
```

---

## 🔗 Flujo del Pipeline MoE

```
01: Adquisición → demand.parquet
    ↓
02: Preprocesamiento → features.parquet
    ↓
03: Clustering Dual → asignaciones + centroides
    ↓
04: Análisis Clusters → clasificadores entrenados
    ↓
05: Selección Expertos → 25 modelos expertos
    ↓
06: Gating Design → mecanismo routing
    ↓
07: HPO → parámetros optimizados
    ↓
08: Ensamble Producción → MoEForecaster.pkl
    ↓
09: Benchmark → MoE vs. 30 modelos
```

---
