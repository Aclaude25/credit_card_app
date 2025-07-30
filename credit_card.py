from math import hypot, gamma

import streamlit as st
import pandas as pd
import numpy as np
from pandas.core.common import random_state
from scipy.constants import precision
from scipy.stats import bootstrap
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Application de machine learning pour la détection de fraude par carte de credit"
    )
    st.subheader("Auteur : Gokou Aimé Claude")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('creditcard.csv')
        return data
    #Affichage de la table de données
    df = load_data()
    df_sample = df.sample(100) #afficher un échantillon de df
    if st.sidebar.checkbox("Afficher les Données brutes", False):
        st.subheader("Jeu de données 'creditcard' : échantillon de 100 obervations")
        st.write(df_sample )

    seed = 123

    #Train/Test Split (diviser les jeux de données en données de test et données d'entrainement)
    @st.cache_data(persist=True)
    def split(df):
        y = df['Class']
        x = df.drop('Class', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.2,
            stratify=y,    #pour regler le problème de déséquilibre de class lors des split des données
            random_state=seed)
        return x_train, x_test, y_train, y_test


    x_train, x_test, y_train, y_test = split(df)

    class_names = ["Transaction Authentique", "Transaction Frauduleuse"]

    classifier = st.sidebar.selectbox(
        "Classificateur",
        ("Random Forest", "SVM", "Logistic Regression")
    )

    #Analyse de la peformance du modèle
    def plot_perf(graphes):
        if "Confusion matrix" in graphes:
            st.subheader("Matrice de confusion")
            plot_confusion_matrix(
                model,
                x_test,
                y_test,
                display_labels= class_names
            )
            st.pyplot()

        if "Roc curve" in graphes:
            st.subheader("Courbe ROC")
            plot_roc_curve(
                model,
                x_test,
                y_test
            )
            st.pyplot()

        if "Precision_Recall curve" in graphes:
            st.subheader("Courbe Precision_Recall ")
            plot_precision_recall_curve(
                model,
                x_test,
                y_test
            )
            st.pyplot()

    # SVM
    if classifier == "SVM":
        st.sidebar.subheader("HyperParamètres du modèle")
        hyp_c = st.sidebar.number_input(
            "Choisir la valeur du paramètre de regularisation",
            0.01, 10.0
        )
        kernel = st.sidebar.radio(
            "Choisir le kernel",
            ("rbf", "linear")
        )
        gamma = st.sidebar.radio(
            "Gamma",
            ("scale", "auto")
        )

        graphe_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du model ML",
            ("Confusion matrix", "Roc curve", "Precision_Recall curve")
        )

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Suport Vector Machine (SVM) Results")

            # Initialisation d'un objet RandomForestClassifier
            model = SVC(
                C=hyp_c,
                kernel=kernel,
                gamma=gamma
            )

            # Entrainement de l'algorithme
            model.fit(x_train, y_train)

            # Prediction
            y_pred = model.predict(x_test)

            # Metriques de performance
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, labels=class_names)
            recall = recall_score(y_test, y_pred, labels=class_names)

            # Afficher les métriques dans l'application
            st.write("Accuracy:", accuracy.round(3))
            st.write("Precision:", precision.round(3))
            st.write("Recall", recall.round(3))

            # Afficher les graphiques de performance
            plot_perf(graphe_perf)

    # Regression logistique
    if classifier == "Logistic Regression":
        st.sidebar.subheader("HyperParamètres du modèle")
        hyp_c = st.sidebar.number_input(
            "Choisir la valeur du paramètre de regularisation",
            0.01, 10.0
        )
        nbre_max_iteraction = st.sidebar.number_input(
            "Choisir le nombre maxumum d'itéraction",
            100, 1000, step=10
        )

        graphe_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du model ML",
            ("Confusion matrix", "Roc curve", "Precision_Recall curve")
        )

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Logistic Regression Results")

            # Initialisation d'un objet RandomForestClassifier
            model = LogisticRegression(
                C=hyp_c,
                max_iter=nbre_max_iteraction,
                random_state= seed
            )

            # Entrainement de l'algorithme
            model.fit(x_train, y_train)

            # Prediction
            y_pred = model.predict(x_test)

            # Metriques de performance
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, labels=class_names)
            recall = recall_score(y_test, y_pred, labels=class_names)

            # Afficher les métriques dans l'application
            st.write("Accuracy:", accuracy.round(3))
            st.write("Precision:", precision.round(3))
            st.write("Recall", recall.round(3))

            # Afficher les graphiques de performance
            plot_perf(graphe_perf)



    #Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("HyperParamètres du modèle")
        n_arbres = st.sidebar.number_input("Choisir le nombre d'arbres dans la forêt",
                                               100, 1000, step=10, key='n_arbres'
        )
        profondeur_arbres = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre",
            1, 20, step=1, key='profondeur_arbres'
        )
        bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la créatiion d'un arbres?",
            ("True", "False"), key='bootstrap'
        )

        graphe_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du model ML",
            ("Confusion matrix", "Roc curve", "Precision_Recall curve")
        )

        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Random Forest Results")

            #Initialisation d'un objet RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=n_arbres,
                max_depth=profondeur_arbres,
                bootstrap=bootstrap,
                random_state=seed
            )

            #Entrainement de l'algorithme
            model.fit(x_train, y_train)

            #Prediction
            y_pred = model.predict(x_test)

            #Metriques de performance
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, labels=class_names)
            recall = recall_score(y_test, y_pred, labels=class_names)

            #Afficher les métriques dans l'application
            st.write("Accuracy:", accuracy.round(3))
            st.write("Precision:", precision.round(3))
            st.write("Recall", recall.round(3))

            #Afficher les graphiques de performance
            plot_perf(graphe_perf)





main()



