# BearsBack

API en Python (FastAPI) que usa modelos de machine learning entrenados con [Orange](https://orangedatamining.com/) para predecir si los daños en un hueso se deben a **depredación** (*Predation*) o a **carroñeo** (*Scavenging*), a partir de sus marcas tafonómicas.

Está pensada como backend de un frontend en React (CORS habilitado para `http://localhost:3000`).

---

## 🦴 Datos

El dataset `damaged.csv` está en formato Orange, con 3 filas de cabecera, y tiene **841 restos óseos**:

| Clase (`Death`) | Registros |
| --- | --- |
| Predation | 519 |
| Scavenging | 322 |

Especies representadas (`Taxa`, columna meta): *Ovis aries* (359), *Equus caballus* (194), *Cervus elaphus* (115), *Capra hircus* (79), *Capreolus capreolus* (64), *Sus scrofa* (16) y *Bos taurus* (14).

### Variables de entrada

| Variable | Tipo | Valores |
| --- | --- | --- |
| `Age` | categórica | `ad` (adulto), `inf` (infantil), `juv` (juvenil), `neo` (neonato), `sen` (senil) |
| `Skeletal_region_2` | categórica | `Appendicular`, `Axial`, `Cranial`, `Girdles`, `Rib`, `V. Cervical`, `V. Lumbar`, `V. Thoracic` |
| `Size` | categórica | `Large`, `Small` |
| `N_Pits`, `N_Scores`, `N_Fractures`, `N_Crushing`, `N_Peeling`, `N_Cren_Edge`, `N_Furrowing` | numérica | Número de marcas de cada tipo |
| `Area_Scores`, `Area_Crushing`, `Area_Crenulated_Edge`, `Area_Furrowing`, `Area_Pits`, `Area_Fracture`, `Area_Peeling` | categórica | Zona del hueso donde aparece la marca: `0` (ninguna), `shaft`, `dist_shaft`, `body`, `neck`, `spinous_process`, `transverse_process`, `other` |

## 🤖 Modelos

Todos están entrenados en Orange y guardados como `.pkcls`:

| Archivo | Modelo |
| --- | --- |
| `Model2.pkcls` | Árbol de decisión |
| `ModelBayes.pkcls` | Naive Bayes |
| `ModelLogisticRegression.pkcls` | Regresión logística |
| `ModelNeural.pkcls` | Red neuronal |
| `ModelForest.pkcls` | Random Forest |
| `ModelGradient.pkcls` | Gradient Boosting |
| `ModelTree.pkcls` | Árbol de decisión (versión alternativa) |
| `Model1.pkcls` | Modelo de prueba: clasifica un animal como ave o anfibio según si vuela (`animals.csv`) |

`animal-tree.ows` es el workflow de Orange del ejemplo de prueba de animales.

## 🚀 Puesta en marcha

### Requisitos

- Python 3.10 o superior

### Instalación

```bash
git clone https://github.com/cosmona/BearsBack.git
cd BearsBack
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

> ⚠️ Los modelos se guardaron con **scikit-learn 1.5.2** y necesitan **PyQt5** para cargarse. `requirements.txt` ya fija esas versiones. Con otra versión de scikit-learn la API no arranca.

### Arrancar

```bash
uvicorn main2:app --reload
```

La API queda en [http://localhost:8000](http://localhost:8000) y la documentación interactiva (Swagger) en [http://localhost:8000/docs](http://localhost:8000/docs).

El ejemplo de animales (`main.py`) se arranca con `uvicorn main:app --reload`.

## 🔌 Endpoints (`main2.py`)

| Método | Ruta | Descripción |
| --- | --- | --- |
| `POST` | `/predict-tree` | Predicción con el árbol de decisión |
| `POST` | `/predict-bayes` | Predicción con Naive Bayes |
| `POST` | `/predict-logisticregression` | Predicción con regresión logística |
| `GET` | `/test-and-score` | Accuracy, Precision, Recall y F1 (respecto a *Scavenging*) del árbol, Bayes y regresión logística sobre `damaged.csv` |
| `GET` | `/tree-viewer` | Árbol de decisión en texto plano: cada nodo muestra `[nº Predation, nº Scavenging]` y su condición |

### Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predict-tree \
  -H "Content-Type: application/json" \
  -d '{
    "Age": "inf",
    "Skeletal_region_2": "Rib",
    "Size": "Small",
    "N_Pits": 0,
    "N_Scores": 1,
    "N_Fractures": 0,
    "N_Crushing": 1,
    "N_Peeling": 0,
    "N_Cren_Edge": 0,
    "N_Furrowing": 0,
    "Area_Scores": "dist_shaft",
    "Area_Crushing": "dist_shaft",
    "Area_Crenulated_Edge": "0",
    "Area_Furrowing": "0",
    "Area_Pits": "0",
    "Area_Fracture": "0",
    "Area_Peeling": "0"
  }'
```

### Respuesta

```json
{ "prediction_death": 0 }
```

| `prediction_death` | Significado |
| --- | --- |
| `0` | Predation (depredación) |
| `1` | Scavenging (carroñeo) |

## 📁 Estructura

```
BearsBack/
├── main2.py            # API principal: predicción de depredación / carroñeo
├── main.py             # API de ejemplo: ave o anfibio
├── requirements.txt    # Dependencias con versiones fijadas
├── damaged.csv         # Dataset de restos óseos (formato Orange)
├── damaged.f.csv       # Copia del dataset
├── animals.csv         # Dataset de ejemplo de animales
├── animal-tree.ows     # Workflow de Orange del ejemplo
└── *.pkcls             # Modelos entrenados con Orange
```

> Los valores categóricos se escriben exactamente como en el dataset, respetando mayúsculas: `Small`, `Rib`, `V. Thoracic`... Si un valor no existe, la API responde `{"error": "Value xx does not exist"}`.

## ⚠️ Limitaciones

- Los modelos de red neuronal, Random Forest y Gradient Boosting se cargan, pero todavía no tienen endpoint.
- `/test-and-score` evalúa sobre los mismos datos con los que se entrenaron los modelos, así que las métricas salen optimistas.
