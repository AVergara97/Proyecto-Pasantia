import LibreriasML
import Preprocesamiento
import NoSupervisado
import EntrenamientoSupervisado
import Validation
import Prediccion

data_preprocesada = Preprocesamiento.preprocesamiento("dataset.csv")
salida_nosupervisado = NoSupervisado.nosupervisado(data_preprocesada)
X, y, X_train, X_val, y_train, y_val, rf_classifier1 = EntrenamientoSupervisado.entrenamiento(salida_nosupervisado)
LibreriasML.joblib.dump(rf_classifier1, 'salida_ML.pkl')
validation = Validation.validation(X, y, X_train, X_val, y_train, y_val, rf_classifier1)
#prediccion = Prediccion.prediccion(salida_supervisado)

