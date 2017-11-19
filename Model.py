from sklearn.svm import SVC

def model(X_train, y_train):
    svc = SVC(kernel="rbf", random_state=42)
    svc.fit(X_train, y_train)

    return svc
