
# Objectif

Le problème de classification des pièces musicales en fonction du genre fut adressé en évaluent les ensembles de primitives, identifier les forces et faiblesse des algorithmes, en analysant le type de prétraitement et technique de réduction de dimensionnalité à employer. Il était question d'établir les méthodes de validation, optimiser les hyperparamètres des modèles et combiner les classificateurs dans le but de maximiser la performance du système.

# Install
Pour installer les packages du projet, exécuter la commande suivante dans le dossier root du projet
~~~~
pip install -e .
~~~~

Pour ajouter un module au projet, spécifiez ce module dans le fichier setup.py

Le script pour effectuer l'inférence ce situe dans /scipts/predict.py
Malheuresement, celui-ci n'est pas fonctionnel, car la sérialisation de certains modèles n'est pas présente dans les fichiers de remise. Cela est dû au fait que ces fichiers de sérialisation dépasse la taile permise par Moodle. Si vous avez vraiment besoin d'exécuter l'inférence, contactez moi à l'adresse marc-antoine.charland.1@ens.etsmtl.ca et on trouvera une façon de vous faire parvenir ces quelques GB de fichiers.

# Requirements

- opencv_contrib_python==3.4.3.18
- numpy==1.15.4
- plotly==3.9.0
- scipy==1.1.0
- astropy==3.1.2
- matplotlib==3.0.0
- Pillow==6.0.0
- scikit_learn==0.21.2
