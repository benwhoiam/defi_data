# Présentation : Panorama des méthodes de classification de texte (Challenges 1 à 8)

---

## 1. Prétraitement linguistique

- **Nettoyage** : suppression du HTML, des URLs, mise en minuscules.
- **Lemmatisation** : réduction des mots à leur racine avec SpaCy.
- **Suppression des stopwords** : élimination des mots fréquents sans valeur informative.
- **Tokenisation** : découpage du texte en unités (mots ou tokens).

---

## 2. Représentations vectorielles du texte

- **Bag-of-Words (BOW)** : chaque texte devient un vecteur d’occurrences de mots.
- **TF-IDF** : pondération des mots selon leur fréquence et leur rareté dans le corpus.
- **Embeddings SpaCy** : chaque mot ou texte est représenté par un vecteur dense pré-entraîné (en_core_web_sm ou en_core_web_lg).
- **Tokenization + Padding** : pour les modèles séquentiels, chaque texte est transformé en séquence d’indices de mots, puis normalisé en longueur.

---

## 3. Méthodes de classification utilisées

### a. Modèles linéaires

- **Régression logistique** (Challenge 5) : modèle linéaire pour la classification multi-classes sur embeddings SpaCy.
- **SVM linéaire (LinearSVC)** (Challenge 6) : séparation des classes par un hyperplan dans l’espace TF-IDF.

### b. Réseaux de neurones

- **MLP (Multi-Layer Perceptron)** (Challenges 2, 4) : réseau de neurones à couches denses, utilisé sur BOW, TF-IDF ou embeddings.
- **LSTM** (Challenge 3) : réseau de neurones récurrent pour exploiter l’ordre des mots dans les séquences.
- **Réseau Keras avec Embedding + GlobalAveragePooling** (Challenge 3) : architecture simple pour la classification multi-classes.

### c. Modèles d’ensemble

- **RandomForestClassifier** (Challenges 7, 8) : ensemble d’arbres de décision, robuste au bruit et aux features non pertinentes.

---

## 4. Pipeline général

1. **Chargement des données** (JSON, CSV)
2. **Prétraitement linguistique** (SpaCy, nettoyage, lemmatisation)
3. **Vectorisation** (BOW, TF-IDF, embeddings, tokenization)
4. **Séparation train/validation** (pour évaluer la généralisation)
5. **Entraînement du modèle** (choix du classifieur)
6. **Évaluation** (classification_report, accuracy, F1-score)
7. **Prédiction sur le test**
8. **Génération du fichier de soumission**

---

## 5. Mathématiques derrière les méthodes

- **TF-IDF** : $tfidf(w, d) = tf(w, d) \\times \\log\\left(\\frac{N}{df(w)}\\right)$
- **Embeddings** : chaque mot/texte est un vecteur dans un espace sémantique appris.
- **SVM** : maximise la marge entre les classes.
- **RandomForest** : vote majoritaire d’arbres de décision.
- **Réseaux de neurones** : apprentissage de fonctions non linéaires pour séparer les classes.

---

## 6. Points forts et limites

- **TF-IDF/BOW** : simple, efficace, mais ignore l’ordre des mots.
- **Embeddings** : capture la sémantique, utile pour modèles linéaires ou réseaux.
- **LSTM** : exploite la séquence, mais plus coûteux.
- **RandomForest** : robuste, peu sensible au surapprentissage, mais moins performant sur données très haute dimension.
- **SVM** : performant sur petits jeux de données, sensible au choix des features.

---

## 7. Conclusion

- Un large éventail de méthodes a été testé, du plus simple (BOW + MLP) au plus avancé (LSTM, embeddings, RandomForest).
- Le choix de la méthode dépend du volume de données, du temps de calcul disponible et de la nature du texte.
- L’importance du prétraitement et de la vectorisation est cruciale pour la performance finale.

---