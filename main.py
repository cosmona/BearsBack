import Orange
import pickle
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Cargar el modelo entrenado con Orange
modelo = pickle.load(open("Model1.pkcls", "rb"))

app = FastAPI()

# Entrada esperada por el modelo
class InputData(BaseModel):
    flies: str  # Indica si el animal vuela o no ("yes"/"no")

@app.post("/predict")
def predict(input: InputData):
    try:
        # Convertir 'flies' a un valor numérico (1 para "yes" y 0 para "no")
        flies_value = 1 if input.flies.lower() == "yes" else 0
        
        # Crear la tabla de entrada solo con el valor de 'flies'
        data = [[flies_value]]  # Solo pasa el valor de flies
        
        # Crear la tabla de Orange usando el dominio del modelo
        table = Orange.data.Table.from_list(modelo.domain, data)
        
        # Realizar la predicción
        prediction = modelo(table)

        # Usar logging para asegurarse de que se vea la predicción
        logging.info(f"Predicción: {prediction}")
        logging.info(f"Predicción (valor numérico): {prediction[0]}")
        
        # Devolver el tipo de animal basado en la predicción
        if prediction[0] == 0:
            result = "Anfibio"
        else:
            result = "Ave"
        
        return {"flies": input.flies, "prediction": result}
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}
