import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier


class Support_vector_machine:

    def fit_predict(self, X_train, X_test, y_train, y_test, kernel, C):
        if kernel == 'rbf':
            gamma=input("Inserire gamma :")
            svm=SVC(kernel=str(kernel), gamma=np.float32(gamma), C=C, random_state=1)
        else:
            svm= SGDClassifier(loss='hinge')
            #svm=SVC(kernel=str(kernel),C=C, random_state=1)
        svm.fit(X_train, y_train)

        y=svm.predict(X_test)
        print("Confusion matrix: ")
        print(confusion_matrix(y, y_test))
        acc=accuracy_score(y, y_test)
        print(f"Accuracy score: {acc}")