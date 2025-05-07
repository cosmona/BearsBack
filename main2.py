import Orange
import pickle
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from Orange.preprocess import Discretize

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado con Orange
modeloTree = pickle.load(open("Model2.pkcls", "rb"))
modeloBayes = pickle.load(open("ModelBayes.pkcls", "rb"))
modeloLogisticRegression = pickle.load(open("ModelLogisticRegression.pkcls", "rb"))
modeloNeural = pickle.load(open("ModelNeural.pkcls", "rb"))
modeloForest = pickle.load(open("ModelForest.pkcls", "rb"))
modeloGradient = pickle.load(open("ModelGradient.pkcls", "rb"))

# Load the dataset
dataset = Orange.data.Table("damaged.csv")

# Discretize the target variable
discretizer = Discretize()
dataset = discretizer(dataset)

app = FastAPI()

# Habilitar CORS para el frontend (React por ejemplo en localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Cambia esto si tu frontend está en otro dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Entrada esperada por el modelo
class InputData(BaseModel):
    Age: str
    Skeletal_region_2: str
    Size: str
    N_Pits: int
    N_Scores: int
    N_Fractures: int
    N_Crushing: int
    N_Peeling: int
    N_Cren_Edge: int
    N_Furrowing: int
    Area_Scores: str
    Area_Crushing: str
    Area_Crenulated_Edge: str
    Area_Furrowing: str
    Area_Pits: str
    Area_Fracture: str
    Area_Peeling: str

@app.post("/predict-tree")
def predict(input: InputData):
    try:
        # Mostrar las categorías del modelo
        logging.debug(f"Categorías del modelo Tree: {modeloTree.domain.variables}")
        for var in modeloTree.domain.variables:
            logging.debug(f"Variable: {var.name}, ¿Es discreta? {var.is_discrete}")

        # Crear la tabla de entrada con los valores
        data = [[
            input.Age,
            input.Skeletal_region_2,
            input.Size,
            input.N_Pits,
            input.N_Scores,
            input.N_Fractures,
            input.N_Crushing,
            input.N_Peeling,
            input.N_Cren_Edge,
            input.N_Furrowing,
            input.Area_Scores,
            input.Area_Crushing,
            input.Area_Crenulated_Edge,
            input.Area_Furrowing,
            input.Area_Pits,
            input.Area_Fracture,
            input.Area_Peeling
        ]]

        # Crear tabla de Orange con el dominio del modelo
        table = Orange.data.Table.from_list(modeloTree.domain, data)
        
        # Realizar la predicción
        prediction = modeloTree(table)
        logging.debug(f"Predicción: {prediction}")
        prediction_value = int(prediction[0])

        logging.info(f"Predicción: {prediction_value}")

       
      
        # Respuesta con la predicción y la ruta del Tree Viewer (si existe)
        return {
            "prediction_death": prediction_value,
        }

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": str(e)}
    
@app.post("/predict-bayes")
def predict(input: InputData):
    try:
        # Mostrar las categorías del modelo
        logging.debug(f"Categorías del modelo Tree: {modeloBayes.domain.variables}")
        for var in modeloBayes.domain.variables:
            logging.debug(f"Variable: {var.name}, ¿Es discreta? {var.is_discrete}")

        # Crear la tabla de entrada con los valores
        data = [[
            input.Age,
            input.Skeletal_region_2,
            input.Size,
            input.N_Pits,
            input.N_Scores,
            input.N_Fractures,
            input.N_Crushing,
            input.N_Peeling,
            input.N_Cren_Edge,
            input.N_Furrowing,
            input.Area_Scores,
            input.Area_Crushing,
            input.Area_Crenulated_Edge,
            input.Area_Furrowing,
            input.Area_Pits,
            input.Area_Fracture,
            input.Area_Peeling
        ]]

        # Crear tabla de Orange con el dominio del modelo
        table = Orange.data.Table.from_list(modeloBayes.domain, data)
        
        # Realizar la predicción
        prediction = modeloBayes(table)
        prediction_value = int(prediction[0])

        logging.info(f"Predicción: {prediction_value}")

       
      
        # Respuesta con la predicción y la ruta del Tree Viewer (si existe)
        return {
            "prediction_death": prediction_value,
        }

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": str(e)}

@app.post("/predict-logisticregression")
def predict_logistic_regression(input: InputData):
    try:
        # Mapear valores categóricos a números
        area_mapping = {
            "0": 0,
            "dist_shaft": 1,
            "other": 2,
            "shaft": 3,
            "spinous_process": 4,
            "transverse_process": 5,
            "body": 6,
            "neck": 7
        }

        age_mapping = {
            "ad": 0,  # Adulto
            "inf": 1,  # Infantil
            "juv": 2,  # Juvenil
            "neo": 3,  # Neonato
            "sen": 4   # Senil
        }

        size_mapping = {
            "large": 1,
            "small": 0
        }

        skeletal_region_mapping = {
            "appendicular": 0,
            "axial": 1,
            "cranial": 2,
            "girdles": 3,
            "rib": 4,
            "v. cervical": 5,
            "v. lumbar": 6,
            "v. thoracic": 7,
            "transverse_process": 8
        }

        # Convertir valores categóricos
        age = age_mapping.get(input.Age.strip().lower(), -1)
        size = size_mapping.get(input.Size.strip().lower(), -1)
        skeletal_region = skeletal_region_mapping.get(input.Skeletal_region_2.strip().lower(), -1)

        # Mapear valores categóricos de las áreas
        area_scores = area_mapping.get(input.Area_Scores.strip().lower(), -1)
        area_crushing = area_mapping.get(input.Area_Crushing.strip().lower(), -1)
        area_crenulated_edge = area_mapping.get(input.Area_Crenulated_Edge.strip().lower(), -1)
        area_furrowing = area_mapping.get(input.Area_Furrowing.strip().lower(), -1)
        area_pits = area_mapping.get(input.Area_Pits.strip().lower(), -1)
        area_fracture = area_mapping.get(input.Area_Fracture.strip().lower(), -1)
        area_peeling = area_mapping.get(input.Area_Peeling.strip().lower(), -1)

        # Validar que los valores categóricos sean válidos
        if age == -1:
            raise ValueError("Valor no válido para la variable 'Age'.")
        if size == -1:
            raise ValueError("Valor no válido para la variable 'Size'.")
        if skeletal_region == -1:
            raise ValueError("Valor no válido para la variable 'Skeletal_region_2'.")
        if -1 in [area_scores, area_crushing, area_crenulated_edge, area_furrowing, area_pits, area_fracture, area_peeling]:
            raise ValueError("Valor no válido para una de las variables de área.")

        # Crear la tabla de entrada con valores predeterminados para las características faltantes
        data = [[
            age,  # Usar el valor numérico mapeado
            skeletal_region,  # Usar el valor numérico mapeado
            size,  # Usar el valor numérico mapeado
            input.N_Pits,
            input.N_Scores,
            input.N_Fractures,
            input.N_Crushing,
            input.N_Peeling,
            input.N_Cren_Edge,
            input.N_Furrowing,
            area_scores,
            area_crushing,
            area_crenulated_edge,
            area_furrowing,
            area_pits,
            area_fracture,
            area_peeling
        ]]

        # Rellenar con valores predeterminados si faltan características
        while len(data[0]) < len(modeloLogisticRegression.domain.attributes):  # Ajustar al número de características esperadas por el modelo
            data[0].append(0)  # Usar 0 como valor predeterminado

        # Crear tabla de Orange con el dominio del modelo
        table = Orange.data.Table.from_list(modeloLogisticRegression.domain, data)

        # Convertir variables categóricas a continuas
        continuizer = DomainContinuizer()
        table = continuizer(table)
        
        # Realizar la predicción
        prediction = modeloLogisticRegression(table)
        prediction_value = int(prediction[0])

        logging.info(f"Predicción: {prediction_value}")

        # Devolver el resultado de la predicción
        return {
            "prediction_death": prediction_value,
        }

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": str(e)}

@app.get("/tree-viewer")
def tree_viewer():
    try:
        # Verificar si el modelo es un árbol de decisión
    
        # Exportar el árbol como una imagen
        tree_image_path = "tree_viewer.png"
        Orange.widgets.visualize.owtreeviewer.OWTreeGraph.save_graph(modelo.tree, tree_image_path)

        # Devolver la imagen como respuesta
        return FileResponse(tree_image_path, media_type="image/png")

    except Exception as e:
        logging.error(f"Error al generar el Tree Viewer: {str(e)}")
        return {"error": str(e)}

@app.get("/test-and-score")
def test_and_score():
    try:
        # Load a dataset (replace 'your_dataset.tab' with your actual dataset file)
        dataset = Orange.data.Table("damaged.csv")

        # Define the models to evaluate
        models = [
            ("Tree", modeloTree),
            ("Bayes", modeloBayes),
            ("Logistic Regression", modeloLogisticRegression)
        ]

        # Perform Test and Score
        logging.debug("Starting Test and Score...")
     
        logging.debug(f"evaluation: {Orange.evaluation.TestOnTrainingData}")
        results = Orange.evaluation.TestOnTrainingData(dataset, [model[1] for model in models])

        # Collect evaluation metrics
        scores = {}
        for i, model in enumerate(models):
            scores[model[0]] = {
                "Accuracy": Orange.evaluation.scoring.CA(results)[i],
                "Precision": Orange.evaluation.scoring.Precision(results)[i],
                "Recall": Orange.evaluation.scoring.Recall(results)[i],
                "F1": Orange.evaluation.scoring.F1(results)[i]
            }

        logging.info(f"Test and Score Results: {scores}")

        # Return the results as a JSON response
        return scores

    except Exception as e:
        logging.error(f"Error during Test and Score: {str(e)}")
        return {"error": str(e)}
