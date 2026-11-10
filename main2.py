import Orange
import pickle
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from Orange.preprocess import Discretize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        # El modelo guarda su dominio original y convierte por dentro las
        # variables categóricas (one-hot), así que se le pasan los valores tal cual
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

        table = Orange.data.Table.from_list(modeloLogisticRegression.original_domain, data)

        # Realizar la predicción
        prediction = modeloLogisticRegression(table)
        prediction_value = int(prediction[0])

        logging.info(f"Predicción: {prediction_value}")

        return {
            "prediction_death": prediction_value,
        }

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": str(e)}

@app.get("/tree-viewer", response_class=PlainTextResponse)
def tree_viewer():
    try:
        # Devuelve el árbol de decisión en texto: cada línea es un nodo con
        # [nº Predation, nº Scavenging] y la condición que lo define
        return modeloTree.print_tree()

    except Exception as e:
        logging.error(f"Error al generar el Tree Viewer: {str(e)}")
        return PlainTextResponse(f"Error: {e}", status_code=500)


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

        # Los modelos ya están entrenados: se predice sobre el dataset y se compara
        # con la clase real. TestOnTrainingData espera algoritmos (learners), no modelos.
        # Precision, Recall y F1 se calculan respecto a la clase "Scavenging".
        y_true = dataset.Y
        scores = {}
        for name, model in models:
            y_pred = model(dataset)
            scores[name] = {
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred)),
                "Recall": float(recall_score(y_true, y_pred)),
                "F1": float(f1_score(y_true, y_pred)),
            }

        logging.info(f"Test and Score Results: {scores}")

        # Return the results as a JSON response
        return scores

    except Exception as e:
        logging.error(f"Error during Test and Score: {str(e)}")
        return {"error": str(e)}
