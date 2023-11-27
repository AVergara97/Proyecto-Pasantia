import LibreriasML

def prediccion(data, rf_classifier1):
    # Asegurándose de que los datos estén en el formato correcto para el modelo
    # Si tu modelo fue entrenado con ciertas columnas, asegúrate de que la nueva data también las tenga.

    # Obtener las probabilidades de cada clase
    probas = rf_classifier1.predict_proba(data)
    
    proba_riesgo = probas[:, 1]  # Ajusta el índice según la clase que te interese
    
    # Añadir las probabilidades al dataframe original
    data['NivelDeRiesgo'] = proba_riesgo * 100  # Multiplicar por 100 para convertir a porcentaje
    
    return data
