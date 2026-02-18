"""
Configuration des chemins du projet.
Tous les modules importent les chemins depuis ici.
"""
import os

# Racine du projet = dossier parent de src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers de données
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Dossiers de sortie
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RAPPORTS_DIR = os.path.join(OUTPUT_DIR, "rapports")

# Dossier documentation
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
