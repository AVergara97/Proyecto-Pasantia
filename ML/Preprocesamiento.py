import LibreriasML

# Uniformizar la columna Renta usando expresiones regulares
def uniform_renta(value):
    if LibreriasML.pd.isna(value): 
        return 'Desconocido'
    if LibreriasML.re.search(r'(0-1000000|\$0\s*[-a]\s*\$300\.?000|\$0\s*-\s*\$300\.?000)', value):
        return '0-300000'   
    if LibreriasML.re.search(r'(\$300\.?001\s*[-a]\s*\$500\.?000|\$300\.?000\s*-\s*\$500\.?000)', value):
        return '300001-500000'    
    if LibreriasML.re.search(r'(\$500\.?001\s*[-a]\s*\$1\.?000\.?000|\$500\.?000\s*-\s*\$1\.?000\.?000|entre_500000_y_1000000)', value):
        return '500001-1000000'    
    if LibreriasML.re.search(r'(Más\s*de\s*\$1\.?000\.?000|>\s*\$1\.?000\.?0000|entre_1000000_y_3000000|entre_1\.000\.000_y_3000000)', value):
        return 'Más de 1000000'   
    return 'Desconocido'

def preprocesamiento(df):
    data = df
    # Quitar columnas
    data = data.drop(["RutComercio", "SubProducto","MontoTransaccion" ,"MonedaOrigen", "MonedaTransaccion", "CodigoMensaje", "IDTransaccion", "FechaProceso", "NumeroTarjeta", "RubroComercio", "CiudadComercio", "CodigoComercio", "ContratoOrigen", "CodigoTransaccion", "DescuentoComision", "CodigoAutorizacion", 'ComisionTransaccion',
        'NumeroReferenciaARN', 'NumeroReferenciaRRN','RazonSocialComercio', 'CondicionTransaccion', 'FechaHoraTransaccion', "MontoOrigen", "CodigoPaisComercio"], axis=1)
    
    # Realizar el strip en la columna 'GlosaTransaccion'
    data['GlosaTransaccion'] = data['GlosaTransaccion'].str.strip()
    # Convertir la columna 'FechaPosteoTransaccion' al tipo de dato datetime
    data['FechaPosteoTransaccion'] = LibreriasML.pd.to_datetime(data['FechaPosteoTransaccion'], format="%Y-%m-%dT%H:%M:%S.%f00000000Z")
    # Extraer el día del mes y agregarlo como una nueva columna
    data['DiaDelMes'] = data['FechaPosteoTransaccion'].dt.day
    #Quitar columna "FechaPosteoTransaccion"
    data = data.drop(["FechaPosteoTransaccion"], axis=1)
    # Convertir NaN a cadena vacía
    data["Renta"] = data["Renta"].fillna("")
    data.dropna(inplace=True)

    data['Renta'] = data['Renta'].apply(uniform_renta)

    # Rellenar valores NaN en RiesgoCliente
    data['RiesgoCliente'].fillna('Desconocido', inplace=True)

    # Aplicar One-Hot Encoding
    data = LibreriasML.pd.get_dummies(data, columns = ['Renta', 'RiesgoCliente'], dtype=int)

    #scaler = LibreriasML.RobustScaler()
    #data['MontoTransaccionPesos'] = scaler.fit_transform(data[['MontoTransaccionPesos']])
    # Aplicar transformación logarítmica
    data['MontoTransaccionPesos'] = LibreriasML.np.log1p(data['MontoTransaccionPesos'])  # log(1 + monto)

    scaler = LibreriasML.MinMaxScaler()
    data['DiaDelMes'] = scaler.fit_transform(data[['DiaDelMes']])

    return data
