import LibreriasML

def entrenamiento(data):
    # Read the reported NumeroContrato file
    reported_df = LibreriasML.pd.read_csv("NumeroContrato_reportados.csv")

    # Extract the reported NumeroContrato values as a list
    reported_numeros = reported_df['NumeroContrato'].tolist()

    # Create a new column "is_reported" based on whether NumeroContrato is reported or not
    data['is_reported'] = data['NumeroContrato'].isin(reported_numeros).astype(int)

    # Split data into features and target
    X = data.drop(['is_reported', 'NumeroContrato'], axis=1)
    y = data['is_reported']

    # Assume X, y are your original data and labels
    X_train, X_val, y_train, y_val = LibreriasML.train_test_split(X, y, test_size=0.4, random_state=42)

    rf_classifier1 = LibreriasML.RandomForestClassifier(min_samples_leaf= 2, min_samples_split=2, max_depth= 20, n_estimators= 100, n_jobs= -1,random_state=42, oob_score=True)
    rf_classifier1.fit(X_train, y_train)

    return X, y, X_train, X_val, y_train, y_val, rf_classifier1