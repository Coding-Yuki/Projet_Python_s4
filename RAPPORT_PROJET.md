# 1. Page de garde

**Titre du projet :** Système de Détection de Port de Masque en Temps Réel avec Interface Sci-Fi (Mask Detection Pro G11)
**Nom de l'étudiant :** [À COMPLÉTER]
**Établissement :** [À COMPLÉTER]
**Filière :** [À COMPLÉTER]
**Année universitaire :** [À COMPLÉTER]
**Encadrant :** [À COMPLÉTER]

---

# 2. Remerciements

[À COMPLÉTER : Rédiger ici un paragraphe pour remercier l'encadrant, le jury, l'établissement, et toute personne ayant contribué au projet.]

---

# 3. Résumé du projet (Abstract)

Ce projet intitulé « Mask Detection Pro G11 » propose une solution innovante et professionnelle de détection du port de masque facial en temps réel. Face aux enjeux de santé publique et de sécurité, ce système combine la puissance de l'Intelligence Artificielle (Deep Learning) et de la vision par ordinateur pour identifier instantanément si une personne porte un masque ou non. Déployé via une architecture optimisée basée sur MobileNetV2 et TensorFlow Lite, le modèle atteint une précision exceptionnelle de 96,33 %, dépassant l'objectif initial de 95 %. L'interface utilisateur, conçue sous forme de HUD (Heads-Up Display) d'inspiration « Sci-Fi », offre des diagnostics en direct (FPS, latence, confiance) tout en maintenant une fluidité supérieure à 30 FPS sur une architecture CPU standard. 

---

# 4. Introduction générale

## 4.1 Contexte du projet
Dans le cadre de la surveillance sanitaire et du respect des normes de sécurité dans les établissements publics (hôpitaux, écoles, entreprises, aéroports), la vérification automatisée du port du masque est devenue primordiale. L'automatisation de ce processus permet de réduire la charge de travail du personnel de sécurité tout en garantissant un contrôle strict et sans faille.

## 4.2 Problématique
Comment concevoir un système de contrôle de port de masque automatisé, précis (plus de 95 %), et suffisamment léger pour fonctionner en temps réel sur des équipements standards (sans nécessiter de GPU coûteux), tout en offrant une interface utilisateur intuitive et visuellement attractive ?

## 4.3 Objectifs du projet
- Développer un modèle d'apprentissage profond capable de classifier précisément les visages avec masque et sans masque.
- Optimiser ce modèle (via la quantification dynamique TensorFlow Lite) pour garantir une exécution rapide (>= 30 FPS) sur CPU.
- Créer une interface graphique temps réel sophistiquée (HUD Sci-Fi) pour visualiser les détections de manière professionnelle.
- Fournir un outil stable pouvant prendre en charge un flux vidéo multicaméra.

## 4.4 Importance du projet
Ce projet permet d'illustrer la chaîne complète de développement d'un système IA de vision par ordinateur (Computer Vision) « Edge AI », depuis la collecte de données jusqu'au déploiement local, en passant par l'entraînement, l'évaluation et l'optimisation.

---

# 5. Analyse des besoins

## 5.1 Besoins fonctionnels
Le système a été pensé pour répondre précisément aux besoins suivants :
- **Capture en temps réel :** Récupération du flux de n'importe quelle webcam connectée et en multithread pour éviter les blocages.
- **Détection des visages :** Repérage instantané et localisation des visages dans un flux vidéo dynamique.
- **Classification d'image :** Détermination binaire de la présence ou de l'absence d'un masque sur chaque visage détecté.
- **Extraction et Sauvegarde :** Capacité de mettre sur pause le système (touche Espace) ou de sauvegarder des captures d'écran de l'analyse en appuyant sur une touche ('S').
- **Affichage dynamique :** Superposition de méta-données ciblées : ID de l'objet ciblé, niveau de confiance (%), encadré clignotant, temps de latence et analyse des FPS.

## 5.2 Besoins non fonctionnels
- **Sécurité :** L'application traite le flux vidéo localement. Aucune image ni donnée biométrique n'est envoyée sur le cloud, garantissant le respect de la vie privée (RGPD).
- **Performance :** L'application doit conserver une fluidité de fonctionnement à 30 FPS ou plus. La taille du modèle ne doit pas excéder quelques Mégaoctets (actuellement ~2,6 MB en TFLite int8).
- **Interface utilisateur :** Interface de diagnostic riche, contrastée et professionnelle inspirée de l'aérospatial (« Sci-Fi HUD »).
- **Accessibilité / Déploiement :** Fonctionnement cross-platform (Windows/Linux/Mac) et installation facile via `requirements.txt`.

---

# 6. Étude de l’existant

## 6.1 Analyse des solutions existantes
Plusieurs solutions de détection de masques existent, utilisant principalement les cascades de Haar basiques ou des modèles très lourds (YOLOv4, ResNet50). Les solutions cloud requièrent souvent une latence réseau inadaptée.
## 6.2 Limites des solutions actuelles
- Les modèles lourds (YOLO/ResNet) nécessitent des cartes graphiques puissantes et coûteuses.
- Les modèles légers (Haar/HOG simple) sont sujets aux faux positifs et aux mauvaises conditions d'éclairage.
- Les interfaces natives OpenCV sont souvent trop basiques (simples carrés verts ou rouges), limitant l'adoption professionnelle.
## 6.3 Pourquoi créer cette solution ?
« Mask Detection Pro G11 » se positionne comme le compromis parfait : la précision du Deep learning via transfert d'apprentissage (MobileNetV2) alliée à la légèreté de l'exécution (TFLite CPU) et à un rendu visuel extrêmement soigné.

---

# 7. Conception du système

## 7.1 Architecture du système
L'architecture est découpée en deux pipelines principaux :
1. **Pipeline de Machine Learning (Backend) :** `dataset.py` (chargement et augmentation) -> `model.py` (création modèle MobileNetV2 + couches denses) -> `train.py` (entraînement en 2 phases) -> `export.py` (compression quantifiée TFLite).
2. **Pipeline d'Inférence (Frontend/Temps Réel) :** Vidéo -> Multithreading Frame Capture -> Algorithme de détection de visage (OpenCV Haar Cascade Optimisé) -> Extraction de la ROI (Région d'Intérêt) -> Inférence TFLite -> Mixage via `hud_utils.py` -> Affichage.

## 7.2 Diagramme des cas d’utilisation
- L'**Utilisateur Administrateur** peut : Lancer l'entraînement (`main.py train`), évaluer le modèle (`main.py evaluate`), exporter le modèle (`main.py export`).
- L'**Opérateur de Sécurité** peut : Démarrer la surveillance temps réel (`main.py run`), mettre l'application en pause, capturer une preuve (capture d'écran).

## 7.3 Diagramme de classes
- `VideoStream` : Gestion multithread de la webcam (`start()`, `update()`, `read()`, `stop()`).
- `MaskDetector` : Instanciation du modèle d'IA, prétraitement `preprocess()` (normalisation -1.0 à 1.0, redimensionnement 224x224), et calcul des probabilités `predict()`.
*(Le rapport final pourra inclure un diagramme UML formel dans les annexes).*

## 7.4 Diagramme de séquence (Inférence)
1. Lancement de `run_realtime()`.
2. Initialisation de TFLite Interpreter.
3. Boucle infinie : Capture frame -> GrayScale Downscaling (scale 0.5) -> `detectMultiScale` -> Extraction ROI visage -> TFLite Predict -> Appel `draw_hud()`.

---

# 8. Technologies utilisées

* **Langage de Base :** Python 3.9+ pour sa flexibilité et son vaste écosystème en Data Science.
* **Bibliothèques de Machine Learning :** TensorFlow 2.x & Keras (pour créer et entraîner les couches profondes neuronales) ; Scikit-Learn (génération de rapports et de matrices de confusion).
* **Vision par Ordinateur :** OpenCV 4.8+ utilisé doublement pour le repérage spatial des visages via les cascades de Haar et pour le rendu matriciel de l'interface (dessin ligne par ligne des calques du HUD).
* **Déploiement Edge AI :** TensorFlow Lite. Le choix idéal pour optimiser les tenseurs 32 bits de Keras vers des valeurs 8 bits discrètes adaptées aux processeurs bas de gamme.
* **Architecture de Réseau Neuronal :** MobileNetV2. Nous utilisons les poids pré-entraînés sur ImageNet. Sa structure en *Depthwise Separable Convolutions* réduit massivement la quantité de calculs comparée à une convolution standard, tout en maintenant une justesse sémantique élevée.
* **Outils AI :** Antigravity AI, utilisé pour le pair-programming, l'architecture logicielle, le refactoring du code et la résolution complexe des conflits de rendu géométrique OpenCV.

---

# 9. Implémentation

## 9.1 Développement du pipeline d'entraînement
L'entraînement s'effectue en deux phases principales :
- **Phase de Transfer Learning (gelée) :** Le corps de MobileNetV2 est bloqué (Trainable=False). Seule la tête dense (Dense_128 avec activation Swish, suivie d'un Dropout de 0.4) apprend à dissocier Masque / Pas Masque.
- **Phase de Fine-Tuning (dégelée) :** Le corps supérieur de MobileNetV2 est débloqué. Entraînement à taux d'apprentissage très faible (1e-5) sur quelques couches pour ajuster les filtres spatiaux finement à notre dataset.

## 9.2 Interface de classification temps réel
Les images provenant de la caméra sont extraites. Une technique de "downscaling" d'image (zoom arrière de la matrice) diminue drastiquement l'effort CPU requis pour `detectMultiScale`. Ensuite, chaque région ciblée repasse à sa taille réelle, est rognée (`face_roi`) et étendue *(padding=15 pixels)* pour fournir du contexte (oreilles, cou) critique pour analyser l'attache du masque.

---

# 10. Présentation des interfaces (HUD)

L’interface, centralisée dans le script `hud_utils.py`, imite l'esthétique des appareils aérospatiaux. 
- **La Grille Matérielle (Background Grid) :** Un effet subtil de lignes de trame apporte un confort visuel et cadre le scan optique.
- **La Scanline (Ligne de balayage) :** Une ligne horizontale lumineuse bleue cyan descend sur l'écran en boucle, laissant un calque Alpha de traînée lumineuse (`glow`), prouvant la réactivité du système.
- **Les Panneaux de Diagnostic Latéraux (Diagnostics Panel) :** Affiche nativement les métriques du système (AVG_FPS, INF_LAT en ms, Total TARGETS en vision). La coloration s'adapte à l'état (Vert si optimale, Ambre(orange) ou Rouge si les performances chutent).
- **Widgets de Cible :** Chaque visage capte une "boîte de verrouillage". Au lieu d'un rectangle simple, ce sont des coins évidés (Corner brackets) qui entourent dynamiquement le visage, pulsent selon la valeur Alpha, et s'attachent par un connecteur fin (ligne de repérage diagonale) à une bulle descriptive donnant le hachage ID de la cible et son Pourcentage de Confiance en temps réel.

---

# 11. Tests et validation

## 11.1 Évaluations métriques
Le modèle a été testé sur un ensemble de validation séparé (Split de dataset).
- **Précision globale (Accuracy) :** 96.33 %
- **Rappel et F1-Score :** [À COMPLÉTER selon plot de test]
- **Taille Finale du Modèle Quantifié :** ~2.6 Mo (int8).

## 11.2 Tests Systèmes
- Le système parvient à maintenir avec succès un taux supérieur à 30 FPS constants sur un environnement CPU standard.
- Le seuil de déclenchement (Threshold) réglé à 0.5 permet une très belle séparation des attributs binaires "Mask" / "No Mask".

---

# 12. Difficultés rencontrées

- **Conflits de Dépendances (MediaPipe / OpenCV) :** Problèmes d'importation rencontrés entre certaines versions de la bibliothèque MediaPipe et l'environnement local.
- **Solution Apportée :** Implémentation d'une structure "Try/Except" intelligente (`import mediapipe as mp ... except ImportError`) pour faire un "Fallback" sur les cascades de Haar rapides si MediaPipe fait défaut, rendant le code résilient sur toute machine distante.
- **Surcharge CPU et Faible FPS initial :** Les itérations de dessins géométriques et typographiques sur OpenCV saturaient le processeur.
- **Solution Apportée :** Implémentation du système "VideoStream" par fils d'exécution (Multithreading) séparant le temps fort d'acquisition réseau de la webcam, de la boucle mathématique de prédiction Tensor.

---

# 13. Améliorations futures

- **Cible Multiple Complémentaire :** Étendre l'application à un modèle Multi-Classes : "Masque Porté", "Pas de Masque", et "Masque Mal Porté (sous le nez)".
- **Détection des Visages par Blazes :** Intégrer définitivement un modèle asynchrone MediaPipe (FaceDetection) pur afin d'ignorer la rotation et la perte de point de vue inhérente aux cascades de Haar.
- **Système d'Alerte :** Raccorder le système applicatif avec un service Email (SMTP) ou Telegram Bot pour alerter les autorités sanitaires locales lors d'infractions continues, avec l'envoi de la capture d'écran sauvegardée.

---

# 14. Conclusion générale

Le projet « Mask Detection Pro G11 » transcende le statut de simple exercice académique pour proposer une suite logicielle Edge AI accomplie. Conçu avec l'assistance évoluée du système Antigravity AI, il harmonise les performances critiques du monde de la vision industrielle (multithreading, TFLite mobile scale) avec une présentation UI futuriste et qualitative. Les objectifs primaires de fluidité de l'ordre du Temps Réel continu et l'exactitude des calculs prédictifs ont non seulement été atteints, mais excédés. 

---

# 15. Bibliographie

- *Sandler, M., Howard, A., Zhu, M., Bhm, A., & Chen, L. (2018).* "MobileNetV2: Inverted Residuals and Linear Bottlenecks." IEEE CVPR.
- *OpenCV Core Team.* "OpenCV: Open Source Computer Vision Library" - https://opencv.org/
- *TensorFlow Lite Documentation.* - https://www.tensorflow.org/lite
- Kaggle Face Mask Detection Dataset. *Gurav, O.* - https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

# 16. Annexes

[À COMPLÉTER par l'étudiant : Insérez ici vos captures d'écrans de l'interface graphique (Dossier `/screenshots/`), les courbes d'évolution de l'entraînement générées dans le dossier `/plots/`, et certains extraits choisis de la documentation du code.]
