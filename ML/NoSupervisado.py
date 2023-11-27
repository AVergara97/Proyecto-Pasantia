import LibreriasML

def nosupervisado(data):
    # Feature selection and dimensionality reduction using PCA
    numerical_columns = ['MontoTransaccionPesos', 'DiaDelMes']
    categorical_columns = ['TipoTransaccion', 'GlosaTransaccion', "Comuna"]
    onehot_columns = ["Renta_0-300000", "Renta_300001-500000", "Renta_500001-1000000", "Renta_Más de 1000000", "RiesgoCliente_ALTO", "RiesgoCliente_BAJO", "RiesgoCliente_MEDIO"]
    
    # Cargar el diccionario de encoders
    loaded_encoders = LibreriasML.joblib.load('all_ordinal_encoders.joblib')
    # Codificación de Variables Categóricas
    for col in categorical_columns:
        try:
            data[col] = loaded_encoders[col].transform(data[col].values.reshape(-1, 1))
        except ValueError as e:
            print(f"Error al transformar la columna {col}: {e}")

    # Apply PCA
    n_pca_components = 2
    pca = LibreriasML.PCA(n_components=n_pca_components)
    pca_result = pca.fit_transform(data[numerical_columns + categorical_columns + onehot_columns])

    # Aplicar K-Means 
    desired_clusters = 4
    kmeans = LibreriasML.KMeans(n_clusters=desired_clusters, random_state=42, n_init=10)
    data['kmeans_labels'] = kmeans.fit_predict(pca_result)

    return data