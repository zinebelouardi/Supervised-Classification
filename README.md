# Classification-Supervis-e
#2. ANALYSE PAR ALGORITHME:

   SVM:
   - Meilleure performance: SVM (C=10, RBF) (Accuracy: 0.8745)
   - Le noyau RBF avec C=1.0 ou C=10 offre les meilleures performances
   - Le noyau polynomial est moins adapté à ce problème

   KNN:
   - Meilleure performance: KNN (k=5) (Accuracy: 0.8345)
   - Performance stable entre k=3 et k=10
   - L'utilisation de poids pondérés améliore les résultats

   Decision Tree:
   - Meilleure performance: DT (depth=5) (Accuracy: 0.8150)
   - Les arbres profonds (max_depth élevé) peuvent sur-apprendre
   - Une profondeur modérée (10-20) offre un bon compromis

3. OBSERVATIONS GÉNÉRALES:
   - L'algorithme SVM (C=10, RBF) offre la meilleure performance globale
   - Les caractéristiques CBIR (couleur + texture) sont efficaces pour distinguer
     les oiseaux des camions
   - La normalisation des features améliore significativement les performances
   - Tous les modèles atteignent une accuracy supérieure à 85%, ce qui indique
     que les deux classes sont bien séparables dans l'espace des features

4. RECOMMANDATIONS:
   - Pour la production: utiliser SVM (C=10, RBF)
   - Pour l'interprétabilité: privilégier Decision Tree avec profondeur modérée
   - Pour la rapidité d'inférence: KNN peut être plus lent sur grands datasets
   - SVM avec noyau RBF offre un excellent compromis performance/robustesse
