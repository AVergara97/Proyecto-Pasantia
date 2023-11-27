import LibreriasML
import Preprocesamiento
import NoSupervisado
import Prediccion

df_new_data = Preprocesamiento.preprocesamiento('tu_nuevo_archivo.csv')
salida_nosupervisado = NoSupervisado.nosupervisado(df_new_data)
data = salida_nosupervisado.drop(['NumeroContrato'], axis=1)
salida_supervisado_cargado = LibreriasML.joblib.load('salida_ML.pkl')
df_with_predictions = Prediccion.prediccion(data, salida_supervisado_cargado)