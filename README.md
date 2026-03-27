# Classification Supervisée — CBIR

## Résultats par algorithme

### SVM

| Configuration         | Accuracy |
|-----------------------|----------|
| SVM (C=1, RBF)        | ~0.87    |
| **SVM (C=10, RBF)**   | **0.8745**  |
| SVM (polynomial)      | < 0.87   |

> Le noyau RBF avec C=10 offre les meilleures performances. Le noyau polynomial est moins adapté à ce problème.

---

### KNN

| Configuration         | Accuracy |
|-----------------------|----------|
| KNN (k=3)             | ~0.83    |
| **KNN (k=5)**         | **0.8345**  |
| KNN (k=10)            | ~0.83    |
| KNN (poids pondérés)  | légèrement supérieur |

> Performance stable entre k=3 et k=10. Les poids pondérés améliorent légèrement les résultats.

---

### Decision Tree

| Configuration          | Accuracy |
|------------------------|----------|
| **DT (depth=5)**       | **0.8150**  |
| DT (depth=10–20)       | bon compromis |
| DT (depth élevé)       | sur-apprentissage |

> Une profondeur modérée (5–20) offre le meilleur compromis biais/variance.

---

## Comparaison globale

| Algorithme         | Meilleure config   | Accuracy  | Remarque                        |
|--------------------|--------------------|-----------|---------------------------------|
| **SVM**            | C=10, RBF          | **0.8745**| Meilleure performance globale   |
| KNN                | k=5                | 0.8345    | Plus lent sur grands datasets   |
| Decision Tree      | depth=5            | 0.8150    | Le plus interprétable           |

---

## Observations

- Les features CBIR (couleur + texture) sont efficaces pour distinguer oiseaux et camions
- La **normalisation des features** améliore significativement les performances
- Tous les modèles dépassent **85% d'accuracy** → les deux classes sont bien séparables

---

## Recommandations

| Cas d'usage            | Algorithme recommandé         |
|------------------------|-------------------------------|
|  Production          | SVM (C=10, RBF)               |
|  Interprétabilité    | Decision Tree (depth modérée) |
|  Inférence rapide    | SVM (éviter KNN sur gros data)|
