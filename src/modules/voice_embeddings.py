"""
==========================================================
  MODULE 3 : EMPREINTE VOCALE (Voice Embeddings)
==========================================================

🎯 Objectif : Créer une "empreinte digitale" unique pour chaque voix
📚 Technologie : Resemblyzer + ChromaDB (Vector DB)

Comment ça marche ?
1. Resemblyzer convertit une voix en VECTEUR (256 nombres)
2. ChromaDB stocke ce vecteur et permet la recherche ultra-rapide
3. On compare les nouveaux vecteurs pour identifier la personne

Mise à jour (Sprint 1.5) :
- Remplacement des fichiers .npy par ChromaDB
- Recherche par similarité vectorielle (Cosine Distance)
==========================================================
"""

import os
import numpy as np

# Importer le gestionnaire de base de données
try:
    from src.modules.vector_db_manager import VectorDBManager
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    print(f"⚠️ Erreur import ChromaDB manager : {e}")

# Vérifier si resemblyzer est installé
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("⚠️ resemblyzer n'est pas installé")
    print("   pip install resemblyzer")


# Initialisation globale du DB Manager (Lazy loading)
_DB_MANAGER = None

def get_db():
    global _DB_MANAGER
    if _DB_MANAGER is None and DB_AVAILABLE:
        try:
            _DB_MANAGER = VectorDBManager()
        except Exception as e:
            print(f"❌ Erreur init ChromaDB: {e}")
    return _DB_MANAGER


def init_voice_encoder():
    """Initialise l'encodeur vocal."""
    if not RESEMBLYZER_AVAILABLE:
        return None
    
    print("📥 Chargement de l'encodeur vocal...")
    encoder = VoiceEncoder()
    print("✅ Encodeur vocal chargé !")
    return encoder


def create_voice_embedding(encoder, audio_path):
    """Crée un embedding (vecteur) à partir d'un fichier audio."""
    if encoder is None:
        return None
    
    print(f"\n🎤 Création de l'empreinte vocale pour : {audio_path}")
    wav = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(wav)
    
    print(f"   ✅ Embedding créé ! (dimension: {embedding.shape[0]})")
    return embedding


def save_voice_profile(embedding, speaker_name):
    """Sauvegarde un profil vocal dans ChromaDB."""
    db = get_db()
    if db:
        db.add_speaker(speaker_name, embedding)
    else:
        print("❌ Impossible de sauvegarder : ChromaDB non dispo")


def key_exists(speaker_name):
    """Vérifie si un locuteur existe déjà (pour compatibilité)."""
    db = get_db()
    if db:
        speakers = db.list_speakers()
        return speaker_name in speakers
    return False


def compare_voices(embedding1, embedding2):
    """
    Compare deux empreintes vocales.
    
    Returns:
        Similarité entre 0 (différent) et 1 (identique)
    """
    # Similarité cosinus
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return float(similarity)


def identify_speaker(encoder, audio_path, threshold=0.75):
    """
    Identifie un locuteur en utilisant la recherche vectorielle ChromaDB.
    """
    # 1. Créer l'embedding de l'audio inconnu
    unknown_embedding = create_voice_embedding(encoder, audio_path)
    if unknown_embedding is None:
        return "INCONNU"
    
    # 2. Chercher dans ChromaDB
    db = get_db()
    if not db:
        return "INCONNU"
    
    print(f"\n🔍 Recherche vectorielle dans ChromaDB...")
    best_match = db.identify_speaker(unknown_embedding, threshold=threshold)
    
    if best_match != "INCONNU":
        print(f"✅ Locuteur identifié : {best_match}")
    else:
        print(f"❓ Locuteur non reconnu")
    
    return best_match


# ===== TEST DIRECT =====
if __name__ == "__main__":
    import sys
    # Ajouter la racine du projet au PYTHONPATH
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import AUDIO_DIR

    print("="*60)
    print("  TEST DU MODULE EMPREINTE VOCALE (Resemblyzer + ChromaDB)")
    print("="*60)
    
    if RESEMBLYZER_AVAILABLE and DB_AVAILABLE:
        encoder = init_voice_encoder()
        
        if encoder:
            # Tester avec les fichiers audio
            test_files = [
                os.path.join(AUDIO_DIR, "test_francais.mp3"),
                os.path.join(AUDIO_DIR, "test_arabe.mp3"),
            ]
            
            # 1. Créer et Sauvegarder (si n'existe pas)
            if os.path.exists(test_files[0]):
                emb = create_voice_embedding(encoder, test_files[0])
                save_voice_profile(emb, "Test_User_FR")
            
            # 2. Identifier
            if os.path.exists(test_files[0]):
                identify_speaker(encoder, test_files[0])
                
            if not any(os.path.exists(f) for f in test_files):
                print(f"\n⚠️ Aucun fichier test trouvé")
                print(f"   → Lance d'abord : python tests/demo_simple.py --test generate")
    else:
        print("\n❌ Manque resemblyzer ou chromadb")
