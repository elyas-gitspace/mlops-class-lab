# TP02 - Réponses (Amin Rachid)

Équipe : Elyas Meziani, Michael Li, Amin Rachid. Dépôt : https://github.com/elyas-gitspace/mlops-class-lab

Elyas a écrit le code du TP02 (modules `churnlab`, `scripts/train.py`, scripts de vérification). De mon côté, j'ai relu ce code, je l'ai relancé sur mon Mac en suivant uniquement le README, et j'ai comparé mes résultats aux siens. C'est cette relecture qui a fait remonter le seul vrai problème du TP (partie 6). Les réponses ci-dessous sont les miennes. Les preuves de ma reproduction sont dans `evidence/tp02/macos/`.

## 1. Un premier modèle qui fonctionne

La tâche est une classification binaire. La classe positive est `Churn = Yes` (codée `1`). `customerID` sert uniquement à identifier les clients et à vérifier le découpage. Il n'est jamais donné au modèle, pas plus que `Churn`.

Le découpage est fait une seule fois, stratifié sur `Churn`, avec la graine `42` : 4 225 clients en train, 1 409 en validation et 1 409 en test, avec à chaque fois environ 26,5 % de départs. L'affectation de chaque client est enregistrée dans `split_allocation.csv`. Sur mon Mac, ce fichier est identique à celui d'Elyas pour les 7 043 clients. Le jeu de test n'a servi à rien dans ce TP.

Avec seulement `tenure` et `MonthlyCharges`, la régression logistique obtient 0,783 d'accuracy en validation, contre 0,735 pour la baseline qui prédit toujours « No ». Les matrices de confusion se lisent `[[TN, FP], [FN, TP]]` (lignes = réalité, colonnes = prédiction, 0 = reste, 1 = part).

| Modèle | Accuracy | Matrice de confusion |
|---|---:|---|
| Baseline `DummyClassifier` (ajustée sur le train) | 0,7346 | `[[1035, 0], [374, 0]]` |
| Régression logistique (2 variables) | 0,7828 | `[[950, 85], [221, 153]]` |

**Que donnerait un système qui prédit toujours « pas de départ » ?** Exactement la baseline : 73,5 % d'accuracy et aucun client sur le départ détecté. C'est le meilleur exemple de ce qu'on avait vu au TP01 avec le déséquilibre des classes.

**Une forte accuracy peut-elle coexister avec zéro churner trouvé ?** Oui, et la baseline le prouve. Avec 26,5 % de positifs, l'accuracy récompense surtout le fait de bien classer ceux qui restent, alors que ce sont ceux qui partent qui intéressent l'équipe.

**Quelles données ont le droit d'influencer le choix du modèle ?** Le train pour apprendre, la validation pour comparer. Le test doit rester intact jusqu'à l'évaluation finale. Sinon, on finit par choisir le modèle qui a eu de la chance sur ce jeu précis.

## 2. Prétraitement et modèle dans une seule Pipeline

Les six variables sont `tenure`, `MonthlyCharges` et `TotalCharges` (numériques), puis `Contract`, `InternetService` et `PaymentMethod` (catégorielles).

Avant la pipeline, une étape de lecture (`parse_features`) convertit les colonnes. Elle n'apprend rien et s'applique de la même façon à l'entraînement et à l'inférence. Les 11 `TotalCharges` vides deviennent des valeurs manquantes. J'ai vérifié qu'il s'agit uniquement de clients avec `tenure = 0`, c'est-à-dire des nouveaux clients qui n'ont pas encore été facturés. En revanche, un texte comme `"abc"` provoque une erreur. Plutôt que d'utiliser directement `pd.to_numeric(errors="coerce")`, qui transformerait silencieusement `"abc"` en `NaN`, le code marque d'abord les vraies cases vides, convertit, puis échoue si une valeur non vide n'a pas pu être lue. C'est exactement la distinction que demande l'énoncé entre une valeur manquante autorisée et une donnée malformée.

Dans la pipeline, un `ColumnTransformer` envoie chaque colonne dans sa branche :

- **Numérique** : `SimpleImputer(strategy="median")` puis `StandardScaler`. Les médianes, moyennes et écarts-types sont appris sur le train.
- **Catégorielle** : `SimpleImputer(strategy="most_frequent")` puis `OneHotEncoder(handle_unknown="ignore")`. La catégorie la plus fréquente et la liste des catégories sont apprises sur le train.

Une catégorie jamais vue à l'entraînement (par exemple `"Three year"`) est encodée par des zéros sur toutes les colonnes de `Contract`. La prédiction ne plante pas, mais cela ne veut pas dire qu'elle est fiable pour ce client.

**Que font `fit`, `transform` et `predict` ?** `fit` apprend à partir des données (médianes, catégories, poids du modèle). `transform` applique ce qui a été appris sans rien modifier. `predict` enchaîne les transformations puis le modèle, toujours sans rien apprendre.

**Que se passe-t-il si on ajuste le scaler sur la validation ?** Des informations de la validation entrent dans le modèle, donc ses scores deviennent trop optimistes. De plus, la validation ne serait plus transformée de la même manière que le train, ce qui fausse la comparaison.

**Une Pipeline empêche-t-elle la fuite de données toute seule ?** Non. Si on appelle `fit` sur train et validation réunis, elle apprendra sur les deux sans rien dire. Elle garantit l'ordre des étapes, pas le choix des données qu'on lui donne.

**Qui ordonne quoi ?** La `Pipeline` ordonne les étapes (prétraiter, puis classer). Le `ColumnTransformer` aiguille les colonnes (numériques d'un côté, catégorielles de l'autre).

## 3. Comparer les modèles équitablement

Les trois modèles et la baseline utilisent les mêmes six variables, le même découpage et les mêmes métriques. Chaque candidat reçoit une pipeline neuve (`build_pipeline` est rappelé à chaque fois), donc aucun ne réutilise l'état appris par un autre.

La métrique principale a été fixée dans `configs/tp02_train.json` avant de lancer la comparaison : l'average precision en validation, calculée à partir de la probabilité de la classe 1. Elle mesure la capacité du modèle à classer les clients qui partent au-dessus de ceux qui restent, sans dépendre d'un seuil, et elle ne se laisse pas tromper par le déséquilibre 73/27 comme l'accuracy. La règle de choix prévoit aussi une tolérance de 0,01 : si un modèle plus simple est à moins de 0,01 du meilleur, c'est lui qu'on garde.

| Modèle | AP (Linux) | AP (mon Mac) | Accuracy | Rappel à 0,5 |
|---|---:|---:|---:|---:|
| Baseline | 0,2654 | 0,2654 | 0,7346 | 0,000 |
| Régression logistique | 0,6475 | 0,6475 | 0,8041 | 0,529 |
| Arbre de décision | 0,6173 | 0,6173 | 0,8027 | 0,602 |
| Random forest | 0,6662 | 0,6665 | 0,7935 (0,7928 chez moi) | 0,487 (0,484 chez moi) |

La random forest est retenue dans les deux cas. Son AP dépasse celle de la régression logistique d'environ 0,019, ce qui est au-dessus de la tolérance. Cela dit, ce n'est pas le modèle qui a la meilleure accuracy (la régression logistique) ni le meilleur rappel au seuil de 0,5 (l'arbre). À ce seuil, elle laisse passer environ la moitié des clients qui partent. Si l'équipe veut surtout les attraper, il faudra choisir le seuil sur la validation plutôt que de garder 0,5. Enfin, on ne peut pas dire que la random forest « bat » le premier modèle grâce à l'algorithme, puisque le nombre de variables est aussi passé de 2 à 6.

**Faut-il changer de découpage pour chaque modèle ?** Non. Les écarts mélangeraient alors l'effet du modèle et l'effet d'un jeu de validation plus ou moins facile. Avec le même découpage, la seule chose qui change, c'est le modèle.

**Le modèle le plus précis est-il celui dont l'équipe a besoin ?** Pas forcément, et c'est le cas ici. La régression logistique a la meilleure accuracy, mais l'équipe cherche avant tout à repérer les départs, donc le classement (AP) et le rappel comptent davantage.

**Si deux candidats sont proches ?** On garde le plus simple. Il est plus rapide, plus facile à expliquer et moins exposé au surapprentissage. C'est exactement le rôle de la tolérance dans la règle.

**Une probabilité de 0,8 est-elle un vrai risque de 80 % ?** Pas automatiquement. Les probabilités d'une random forest sont des moyennes de votes d'arbres, pas des probabilités calibrées. Pour le vérifier, il faudrait une courbe de calibration.

## 4. Sauvegarder toute la pipeline entraînée

Le fichier `pipeline.joblib` contient le prétraitement et le classifieur ajustés, dans un seul objet. À côté, `metadata.json` enregistre le hash des données, la graine et l'affectation des clients, le rôle des variables, la politique de prétraitement, les paramètres du modèle, les métriques, la classe positive, le seuil, le commit Git et les versions des paquets. Il contient aussi le SHA-256 de l'artefact, vérifié avant chaque chargement.

Sur mon Mac, `scripts/check_reload.py` recharge l'artefact dans un processus séparé et prédit 25 clients de validation sans colonne `Churn`. L'écart maximal avec les scores calculés juste après l'entraînement est de `1,1e-16` (tolérance `1e-12`), et les 25 prédictions de classe sont identiques. Aucun `fit` n'est appelé dans ce processus.

**Pourquoi sauvegarder seulement le classifieur ne suffit pas ?** Le classifieur attend des nombres mis à l'échelle et des colonnes one-hot. Sans le prétraitement ajusté, il faudrait le reconstruire à la main, avec le risque d'obtenir d'autres médianes ou un autre ordre de colonnes.

**Qu'y a-t-il dans un imputer ou un encoder ajusté ?** Les médianes du train, la catégorie la plus fréquente de chaque colonne et la liste des catégories, qui fixe l'ordre des colonnes one-hot.

**Deux fichiers `model.joblib` peuvent-ils contenir des modèles différents ?** Oui, et on en a eu un exemple concret. Mon `pipeline.joblib` (SHA-256 `e42ab3f7…`) n'est pas le même que celui d'Elyas (`289cd06d…`), alors que le code, la config et les données sont identiques. C'est le même raisonnement qu'au TP01 avec le CSV : le nom du fichier ne prouve rien, le hash si.

**Pipeline sklearn ou pipeline Kubeflow ?** Une `Pipeline` sklearn est un seul objet Python qui enchaîne des transformations et un modèle en mémoire. Une pipeline Kubeflow est un enchaînement d'étapes séparées (un graphe), chacune exécutée dans son propre conteneur, qui se passent des fichiers (par exemple « télécharger, entraîner, évaluer, déployer »).

## 5. Vérifications de fiabilité

J'ai relancé `scripts/check_pipeline.py` sur mon Mac. Les 11 vérifications passent :

- **Découpage** : aucun `customerID` commun entre train, validation et test.
- **Statistiques apprises sur le train seulement** : les médianes de l'imputer et les moyennes du scaler sont celles du train, pas celles du jeu complet.
- **`predict` ne modifie rien** : les statistiques apprises sont identiques avant et après une prédiction.
- **Valeurs manquantes autorisées** : un `TotalCharges` vide, `None` ou `NaN` est remplacé par la médiane du train, et un `Contract` manquant par la catégorie la plus fréquente.
- **Catégorie inconnue** : `"Three year"` est encodée par des zéros, sans erreur.
- **Colonne manquante** : l'absence de `Contract` provoque une erreur visible, plutôt qu'une prédiction silencieusement fausse.
- **Texte invalide** : `"abc"` dans une colonne numérique provoque une erreur, au lieu de devenir une valeur manquante.

Les 16 tests unitaires passent aussi (`uv run pytest`).

## 6. Relecture et problème trouvé

L'énoncé demande d'échanger le projet avec un pair et de noter un problème. J'ai suivi uniquement le README d'Elyas sur une autre plateforme (macOS arm64, contre Linux aarch64 pour lui), avec le même `uv.lock`, la même config et la même graine. Le découpage, la baseline, la régression logistique et l'arbre de décision donnent exactement ses chiffres.

La random forest, elle, ne donne pas tout à fait le même résultat. L'AP passe de 0,6662 à 0,6665, et un client de validation change de prédiction (193 faux négatifs au lieu de 192). Les versions des paquets sont pourtant identiques. La différence vient des calculs en virgule flottante, qui ne sont pas strictement les mêmes d'une plateforme à l'autre, et un modèle composé de 300 arbres y est plus sensible qu'une régression logistique. Le problème n'était pas tant l'écart lui-même, qui ne change pas la sélection, que la façon dont il était présenté. Les réponses d'Elyas indiquaient que relancer `train.py` redonnait les mêmes métriques, et le README ne disait rien de la plateforme. C'est vrai sur une même machine, pas d'une machine à l'autre.

La résolution a été de le préciser dans le README, dans la partie sur ce qui n'est pas encore reproduit, et d'ajouter mes résultats macOS dans `evidence/tp02/macos/` pour que l'écart soit vérifiable. J'en retiens que figer les versions des paquets ne suffit pas à figer les résultats : la plateforme fait aussi partie de l'environnement.

## Ce qui reste à faire

Le jeu de test n'a pas encore été évalué. Il est réservé pour une évaluation unique, une fois la sélection terminée. Le seuil est resté à 0,5 et les probabilités ne sont pas calibrées. La comparaison repose sur un seul découpage de validation, sans validation croisée. Enfin, chaque entraînement est enregistré dans un simple fichier JSON. Le suivi des expériences et le versionnage des modèles arrivent au TP suivant.
