# TP01 - Réponses (Amin Rachid)

Équipe : Elyas Meziani, Michael Li, Amin Rachid. Dépôt : https://github.com/elyas-gitspace/mlops-class-lab

Elyas a monté le projet de départ et la PR sur le README. De mon côté, j'ai repris le projet sur mon Mac pour vérifier qu'il se reproduisait sur une autre machine (partie D), et j'ai rédigé les réponses ci-dessous.

## Partie A - Le projet et les données

**A1. Pourquoi le CSV brut, un modèle entraîné et le code n'ont-ils pas le même rôle dans le dépôt ?**

Ces trois éléments n'ont pas la même origine, et c'est ce qui détermine où ils doivent vivre. Le code, c'est ce qu'on écrit nous-mêmes. Il est petit, il se relit en PR et il a donc toute sa place dans Git. Le CSV brut, c'est ce qu'on reçoit. On ne doit jamais le modifier, et plutôt que de le committer, on l'identifie par son URL source et son empreinte SHA-256 (enregistrées dans `data/raw/telco.source.json`). Enfin, le modèle, c'est ce qu'on produit à partir du code, des données et d'une configuration. Il peut être régénéré, et il ira dans un stockage d'artefacts dans les prochains TP, pas dans Git. Mélanger les trois revient à ne plus savoir ce qui a été écrit, reçu ou généré.

**A2. Pourquoi inclure `Churn` dans les entrées du modèle invaliderait-il la tâche ?**

`Churn` est exactement la colonne qu'on cherche à prédire. Si on la donne au modèle, il n'a plus rien à apprendre : il recopie la réponse. Cela donnerait un score presque parfait en évaluation, alors qu'en production la valeur de `Churn` n'existe pas encore au moment où l'on fait la prédiction. C'est ce qu'on appelle une fuite de la cible (« target leakage »), et c'est le genre d'erreur qui ne se voit pas dans les métriques mais rend le modèle inutilisable.

## Partie B - L'environnement Python

1. **Contrainte Python** : dans `pyproject.toml`, `requires-python = ">=3.11,<3.13"`. **Dépendance directe** : `numpy~=1.26.0`.
2. **La même dépendance dans `uv.lock`** : numpy y est figé en version `1.26.4`, avec l'URL et le hash de chaque fichier téléchargeable (« wheel »).
3. **Dépendance transitive** : `uv tree` montre `scipy 1.17.1` sous `scikit-learn`. On ne l'a jamais déclarée nous-mêmes. C'est scikit-learn qui en a besoin. C'est la même chose pour `python-dateutil` et `tzdata`, tirées par `pandas`.
4. **`pyproject.toml`, `uv.lock` et `.venv/`** : `pyproject.toml` dit ce dont le projet a besoin, avec des plages de versions autorisées. `uv.lock` dit ce qui a réellement été choisi dans ces plages, pour toutes les dépendances (directes et transitives). `.venv/` est l'installation concrète sur une machine donnée. Il se régénère à partir du lock et ne se committe jamais.

**B3.** `numpy ~= 1.26.0` accepte n'importe quelle version `1.26.x`, alors que `numpy==1.26.4` n'en accepte qu'une seule. En pratique, la plage va dans `pyproject.toml` et la version exacte retenue est notée dans `uv.lock`. Une plage laisse de la marge pour les correctifs. Le lock garantit que tout le monde installe la même chose.

## Partie C - Les données Telco

1. **Pourquoi une classification ?** La cible `Churn` ne prend que deux valeurs (`Yes` ou `No`). On prédit une catégorie, pas une quantité.
2. **Que dit le déséquilibre des classes sur l'accuracy ?** Seulement 26,54 % des clients partent (1 869 sur 7 043). Un modèle qui répondrait systématiquement « No » obtiendrait déjà 73,5 % d'accuracy sans détecter un seul client sur le départ. L'accuracy seule ne suffira donc pas. Il faudra regarder la précision, le rappel, le F1 ou l'AUC.
3. **Pourquoi `TotalCharges` est lue comme du texte ?** Onze lignes contiennent un espace (`" "`) à la place d'un nombre. Il suffit d'une seule valeur non numérique pour que pandas lise toute la colonne en `object`, même si elle représente un montant.
4. **Pourquoi `isna()` ne suffit pas ?** Pour pandas, `" "` n'est pas une valeur manquante. C'est une chaîne de caractères non vide. Il faut donc la chercher explicitement avec `.str.strip().eq("")`, ou convertir la colonne avec `pd.to_numeric(..., errors="coerce")` pour que ces cases deviennent des `NaN`.
5. **Quelle preuve montre qu'on a inspecté le même fichier ?** Le SHA-256 (`16320c9c…5e91`), associé à l'URL source figée sur un commit IBM précis. Le nom du fichier ne prouve rien, et on l'a vu concrètement. Le fichier `datasets.csv` qu'on avait reçu contenait exactement les mêmes 7 043 lignes, mais il avait été réenregistré (BOM au début, fins de ligne Windows, chaque ligne entre guillemets). Son hash était donc `62ca518b…` et le script d'inspection l'a refusé. J'en retiens qu'un même contenu « à l'œil » ne veut pas dire les mêmes octets.

## Partie D - Recréation de l'environnement sur macOS

Elyas avait déjà produit les preuves sur sa machine Linux. Plutôt que de refaire la même chose au même endroit, j'ai refait la partie D sur mon Mac pour vérifier que le projet se reproduisait vraiment ailleurs. Les fichiers sont dans `evidence/tp01/macos/`.

Concrètement, j'ai enregistré l'environnement dans `environment-before.json`, renommé `.venv/` en `.venv-backup/`, puis relancé `uv sync --locked` et les deux scripts. L'environnement recréé est identique à celui d'avant (Python 3.11.15, pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, churnlab 0.1.0). `uv.lock` et `pyproject.toml` n'ont pas bougé, et `uv run pytest` passe. Le rapport de données est identique à celui d'Elyas au caractère près (même hash, 7 043 lignes, 5 174 « No » et 1 869 « Yes », aucun doublon, 11 `TotalCharges` vides).

La comparaison avec Linux est plus intéressante. Les versions des paquets sont les mêmes, mais pas le système (macOS arm64 contre Linux aarch64), ni la version exacte de Python (3.11.15 contre 3.11.16), ni celle de uv (0.11.14 contre 0.12.13). Cela montre bien que le lock fige les paquets, pas l'interpréteur ni le système d'exploitation.

| Élément | Preuve produite dans le TP01 | Limite restante |
|---|---|---|
| Code | Commit Git | Les modifications non commitées ne sont pas couvertes par ce commit |
| Choix des paquets | `pyproject.toml`, `uv.lock` | La plateforme et l'interpréteur peuvent changer l'installation (3.11.15 chez moi, 3.11.16 chez Elyas) |
| Environnement installé | Relevés avant/après (Linux et macOS) | Ils ne capturent pas tous les détails du système ou du matériel |
| Données source | URL source et SHA-256 | Cela ne garantit ni la qualité des labels ni l'absence de fuite |
| Modèle entraîné | Pas encore produit | Il faudra des preuves d'entraînement, de configuration et d'artefacts dans les prochains TP |

## Partie E - Travail par pull request

- **PR** : https://github.com/elyas-gitspace/mlops-class-lab/pull/1 (`docs/telco-onboarding` vers `main`).
- **Commits** : `b60315c` (projet initial), `bba201c` (modification du README), `7c7cdfd` (merge dans `main`).
- **Question du relecteur** (@PraiseTheLemon) : « Was the onboarding process not reproducible before? » Pas complètement. Les paquets étaient déjà figés par `pyproject.toml` et `uv.lock`, mais le README ne disait ni d'où venaient les données, ni quels octets attendre, ni comment les vérifier. La PR ajoute l'URL figée, le SHA-256, les commandes d'inspection avec les résultats attendus, la colonne cible et le rôle des dossiers.
- **Vérification sur une autre machine** : ma branche `evidence/macos-reproduction` ajoute les preuves de la partie D.

## Partie F - Restaurer ou retirer de l'index

La différence tient à ce qu'on perd. `git restore README.md` (F1) remet le fichier dans l'état du dernier commit : la modification non indexée est perdue, et Git ne peut pas la récupérer puisqu'elle n'a jamais été enregistrée. `git restore --staged README.md` (F2) ne touche pas au fichier. Il retire seulement la modification de l'index (la « staging area »), si bien qu'elle reste dans le fichier mais ne fera pas partie du prochain commit. Le premier efface donc du travail, le second ne fait que changer ce qui sera commité.

## Ce que le TP01 ne reproduit pas encore

Il n'y a pour l'instant ni modèle, ni configuration d'entraînement, ni graine aléatoire, ni métriques. Le système, la version exacte de l'interpréteur et le matériel ne sont pas figés, comme l'a montré la comparaison entre mon Mac et la machine d'Elyas. Enfin, le hash prouve quels octets on a utilisés, pas que les labels sont justes. Cette base est nécessaire, mais elle ne garantit pas encore qu'un entraînement donnera deux fois le même résultat.
