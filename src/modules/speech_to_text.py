"""
==========================================================
  MODULE 1 : SPEECH-TO-TEXT avec OpenAI Whisper
==========================================================

🎯 Objectif : Convertir un fichier audio en texte
📚 Technologie : OpenAI Whisper (modèle open-source)

Comment ça marche ?
- Whisper est un modèle de Deep Learning (Transformer)
- Il a été entraîné sur 680,000 heures d'audio multilingue
- Il supporte : Français, Arabe, Anglais, et +90 langues
- Il fonctionne EN LOCAL (pas besoin d'internet)

Tailles de modèle disponibles :
  tiny   → 39M params  → Très rapide, moins précis
  base   → 74M params  → Bon compromis pour les tests
  small  → 244M params → Bonne précision
  medium → 769M params → Très bonne précision
  large  → 1550M params → Meilleure précision (lent sur CPU)
==========================================================
"""

import whisper
import os
import json
from datetime import timedelta


def load_model(model_size="base"):
    """
    Charge le modèle Whisper.
    
    Args:
        model_size: Taille du modèle ("tiny", "base", "small", "medium", "large")
    
    Returns:
        Le modèle Whisper chargé
    """
    print(f"📥 Chargement du modèle Whisper '{model_size}'...")
    print(f"   (La première fois, le modèle sera téléchargé automatiquement)")
    model = whisper.load_model(model_size)
    print(f"✅ Modèle '{model_size}' chargé avec succès !")
    return model


def transcribe_audio(model, audio_path, language=None):
    """
    Transcrit un fichier audio en texte.
    
    Args:
        model: Le modèle Whisper chargé
        audio_path: Chemin vers le fichier audio (.wav, .mp3, etc.)
        language: Langue forcée ("fr" pour français, "ar" pour arabe). 
                  None = détection automatique
    
    Returns:
        dict avec : text, language, segments
    """
    if not os.path.exists(audio_path):
        print(f"❌ Fichier non trouvé : {audio_path}")
        return None
    
    print(f"\n🎤 Transcription de : {audio_path}")
    print(f"   Langue : {'Auto-détection' if language is None else language}")
    print(f"   Patience... (ça peut prendre un moment sur CPU)")
    
    # Options de transcription
    options = {}
    if language:
        options["language"] = language
    
    # Transcrire !
    result = model.transcribe(audio_path, **options)
    
    # Résultats
    detected_lang = result.get("language", "inconnu")
    text = result["text"]
    segments = result["segments"]
    
    print(f"\n{'='*60}")
    print(f"📝 RÉSULTAT DE LA TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"🌍 Langue détectée : {detected_lang}")
    print(f"📄 Texte complet :")
    print(f"   {text}")
    print(f"\n⏱️ Segments détaillés :")
    
    for seg in segments:
        start = str(timedelta(seconds=int(seg["start"])))
        end = str(timedelta(seconds=int(seg["end"])))
        print(f"   [{start} → {end}] {seg['text'].strip()}")
    
    print(f"{'='*60}")
    
    return {
        "text": text,
        "language": detected_lang,
        "segments": segments
    }


def transcribe_with_translation(model, audio_path):
    """
    Transcrit ET traduit l'audio en anglais.
    Utile pour vérifier la compréhension de la darija/arabe.
    """
    print(f"\n🔄 Transcription + Traduction en anglais...")
    result = model.transcribe(audio_path, task="translate")
    
    print(f"🇬🇧 Traduction anglaise : {result['text']}")
    return result


# ===== TEST DIRECT =====
if __name__ == "__main__":
    import sys
    # Ajouter la racine du projet au PYTHONPATH
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import AUDIO_DIR

    print("="*60)
    print("  TEST DU MODULE SPEECH-TO-TEXT (Whisper)")
    print("="*60)
    
    # Vérifier que Whisper est installé
    print(f"\n✅ Whisper importé avec succès !")
    print(f"   Version : {whisper.__version__ if hasattr(whisper, '__version__') else 'OK'}")
    
    # Charger le modèle
    model = load_model("base")
    
    # Tester avec les fichiers audio de test
    test_files = [
        (os.path.join(AUDIO_DIR, "test_francais.mp3"), "fr"),
        (os.path.join(AUDIO_DIR, "test_arabe.mp3"), "ar"),
    ]
    
    for audio_file, lang in test_files:
        if os.path.exists(audio_file):
            print(f"\n{'='*60}")
            print(f"  Test avec : {audio_file} (langue: {lang})")
            print(f"{'='*60}")
            result = transcribe_audio(model, audio_file, language=lang)
        else:
            print(f"\n⚠️ Fichier test non trouvé : {audio_file}")
            print(f"   → Lance d'abord : python src/utils/generate_test_audio.py")

