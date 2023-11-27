import LibreriasML

def validation(X, y, X_train, X_val, y_train, y_val, rf_classifier1):
    # Muestra el error OOB
    oob_error = 1 - rf_classifier1.oob_score_
    print("OOB Error: ", oob_error)

    print("RF Train F1:", LibreriasML.f1_score(y_train, rf_classifier1.predict(X_train)))
    print("RF Validation F1:", LibreriasML.f1_score(y_val, rf_classifier1.predict(X_val)))

    # Realiza una validación cruzada de 5 pliegues
    cv_scores = LibreriasML.cross_val_score(rf_classifier1, X, y, cv=5, scoring='f1_macro')

    # Imprime los resultados
    print(f"Puntuaciones F1 de la validación cruzada: {cv_scores}")
    print(f"Puntuación F1 media de la validación cruzada: {LibreriasML.np.mean(cv_scores)}")
    
    return 0