import LibreriasML
import Preprocesamiento
import NoSupervisado
import Prediccion
import pandas as pn
from sql import *

#Carga datos base
df_clientes = fetch_table('DATA_CLIENTES_TAPP')
df_clientes['RutCliente'] = df_clientes['RutCliente'].str.strip()

df_cuentas = fetch_table('DATA_CUENTA_CLIENTE_TAPP')
df_cuentas['RutTitular'] = df_cuentas['RutTitular'].str.strip()
df_cuentas['NumeroContrato'] = df_cuentas['NumeroContrato'].str.strip()
df_cuentas.drop_duplicates(subset=['RutTitular'],inplace=True)

df_transacciones = fetch_table('DATA_TRANSACCION_CLIENTE_TAPP')

#Armar formato de data para el modelo
df_clientes = df_clientes.merge(right=df_cuentas,left_on='RutCliente',right_on='RutTitular',how='inner')
df_transacciones = df_transacciones.merge(right=df_clientes,left_on='NumeroContrato',right_on='NumeroContrato',how='inner')
df_transacciones.drop(['RutTitular','RutCliente'],axis='columns',inplace=True)

#Preprocesamiento
df_transacciones = df_transacciones.astype({'MontoTransaccionPesos':float})
df_prep = Preprocesamiento.preprocesamiento(df_transacciones)

#Agregar columnas de modelo NS
df_prep = NoSupervisado.nosupervisado(df_prep)
df_prep.drop(['NumeroContrato'], axis=1,inplace=True)

df_prep = df_prep.reindex(columns=['Comuna', 'TipoTransaccion', 'GlosaTransaccion',
       'MontoTransaccionPesos', 'DiaDelMes', 'Renta_0-300000',
       'Renta_300001-500000', 'Renta_500001-1000000',
       'Renta_Más de 1000000', 'RiesgoCliente_ALTO', 'RiesgoCliente_BAJO',
       'RiesgoCliente_MEDIO', 'kmeans_labels'])

#Carga de modelo y predicción
model = LibreriasML.joblib.load('salidaP.pkl')
df_predic = Prediccion.prediccion(df_prep,model)

#Merge predictions a la data
df_predic = df_predic.filter(['NivelDeRiesgo'])
data_complete = pn.merge(df_transacciones, df_predic, left_index=True , right_index= True)
data_complete = data_complete.astype({'MontoTransaccionPesos':float})

#Resultados 
df_results = data_complete.groupby('NumeroContrato',as_index = False).agg(
    factor_riesgo=pn.NamedAgg(column="NivelDeRiesgo", aggfunc="mean"),
    numero_transacciones=pn.NamedAgg(column="NivelDeRiesgo", aggfunc="count"),
    monto_total=pn.NamedAgg(column="MontoTransaccionPesos", aggfunc="sum"),
    monto_promedio=pn.NamedAgg(column="MontoTransaccionPesos", aggfunc="mean"),
    monto_maximo=pn.NamedAgg(column="MontoTransaccionPesos", aggfunc="max"))

#Escritura Resultados
data_complete.columns = data_complete.columns.str.lower()
write_lista(data_complete,"transacciones_ml")
df_results = df_results.rename(columns={"NumeroContrato": "numerocontrato"})
write_lista(df_results,"resultados_ml")

print("---END---")




