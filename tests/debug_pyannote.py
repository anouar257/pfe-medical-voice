"""
==========================================================
  DEBUG PYANNOTE — Test d'accès Hugging Face & diarization
==========================================================

🎯 Objectif : Vérifier que l'authentification Hugging Face
              et le chargement du pipeline pyannote fonctionnent.

Usage :
  python tests/debug_pyannote.py

Prérequis :
  - Token HF défini : $env:HF_TOKEN='ton_token' (PowerShell)
  - Conditions acceptées sur les modèles pyannote (voir docs/)
==========================================================
"""

import os
import sys
import traceback

# Configurer l'encodage UTF-8 pour les emojis
sys.stdout.reconfigure(encoding='utf-8')

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import AUDIO_DIR


def run_debug():
    """
    Exécute tous les tests de debug pour pyannote.
    Vérifie : token → login → versions → pipeline → diarization.
    """
    print("=" * 60)
    print("  DEBUG PYANNOTE — Vérification complète")
    print("=" * 60)

    # --- Étape 1 : Vérifier le token ---
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\n❌ Token HF_TOKEN non trouvé dans l'environnement.")
        print("   → Définis-le : $env:HF_TOKEN='ton_token' (PowerShell)")
        return
    print(f"\n✅ Token HF_TOKEN trouvé ({len(token)} caractères)")

    # --- Étape 2 : Login Hugging Face ---
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ Login HuggingFace réussi")
    except Exception as e:
        print(f"❌ Échec du login : {e}")
        return

    # --- Étape 3 : Vérifier les versions ---
    try:
        import pyannote.audio
        import torch
        import torchaudio
        import numpy as np
        print(f"\n📦 Versions installées :")
        print(f"   pyannote.audio : {pyannote.audio.__version__}")
        print(f"   torch          : {torch.__version__}")
        print(f"   torchaudio     : {torchaudio.__version__}")
        print(f"   numpy          : {np.__version__}")
    except ImportError as e:
        print(f"❌ Module manquant : {e}")
        return

    # --- Étape 4 : Charger le pipeline ---
    print("\n=== Chargement du pipeline ===")
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("✅ Pipeline chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur de chargement :")
        traceback.print_exc()
        return

    # --- Étape 5 : Tester la diarization ---
    audio_file = os.path.join(AUDIO_DIR, "test_dialogue.mp3")
    if os.path.exists(audio_file):
        print(f"\n=== Diarization de {audio_file} ===")
        try:
            diarization = pipeline(audio_file, num_speakers=2)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"  [{turn.start:.1f}s → {turn.end:.1f}s] {speaker}")
            print("\n✅ DIARIZATION RÉUSSIE !")
        except Exception as e:
            print(f"❌ Erreur de diarization :")
            traceback.print_exc()
    else:
        print(f"\n⚠️ Fichier non trouvé : {audio_file}")
        print("   → Lance d'abord : python tests/demo_simple.py --test generate")


if __name__ == "__main__":
    run_debug()
