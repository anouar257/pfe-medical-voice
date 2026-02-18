"""
==========================================================
  PIPELINE COMPLET : Audio → Texte → Analyse → Rapport
==========================================================

🎯 Script principal qui orchestre tout le pipeline médical

Étapes :
1. Charger un fichier audio
2. Transcrire avec Whisper (Speech-to-Text)
3. Identifier les locuteurs avec pyannote (si disponible)
4. Analyser le texte (NLP) : symptômes, urgence, hypothèses
5. Générer le rapport structuré en 3 parties

Usage :
  python main.py --audio data/audio/test_dialogue.mp3
  python main.py --audio data/audio/test_francais.mp3 --lang fr
  python main.py --audio data/audio/test_arabe.mp3 --lang ar
==========================================================
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Ajouter le dossier racine au PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RAPPORTS_DIR, AUDIO_DIR


def run_pipeline(audio_path, language=None, num_speakers=2, whisper_model="base"):
    """
    Exécute le pipeline complet.
    
    Args:
        audio_path: Chemin vers le fichier audio
        language: Langue forcée ("fr", "ar") ou None pour auto-détection
        num_speakers: Nombre de locuteurs attendus
        whisper_model: Taille du modèle Whisper
    """
    print("="*60)
    print("  🏥 PIPELINE MÉDICAL - ANALYSE VOCALE")
    print("="*60)
    print(f"  📁 Audio    : {audio_path}")
    print(f"  🌍 Langue   : {'Auto' if language is None else language}")
    print(f"  🗣️ Locuteurs: {num_speakers}")
    print(f"  ⏰ Date     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"\n❌ Fichier non trouvé : {audio_path}")
        print(f"   Génère des fichiers test avec : python -m src.utils.generate_test_audio")
        return
    
    # =============================================
    # ÉTAPE 1 : TRANSCRIPTION (Whisper)
    # =============================================
    print(f"\n{'─'*60}")
    print(f"📌 ÉTAPE 1 : TRANSCRIPTION (OpenAI Whisper)")
    print(f"{'─'*60}")
    
    from src.modules.speech_to_text import load_model, transcribe_audio
    
    model = load_model(whisper_model)
    transcription = transcribe_audio(model, audio_path, language=language)
    
    if not transcription:
        print("❌ La transcription a échoué")
        return
    
    # =============================================
    # ÉTAPE 2 : DIARIZATION (pyannote) - Optionnel
    # =============================================
    diarization_segments = None
    dialogue = None
    
    try:
        from src.modules.speaker_diarization import (
            load_diarization_pipeline, 
            diarize_audio,
            combine_transcription_diarization
        )
        
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print(f"\n{'─'*60}")
            print(f"📌 ÉTAPE 2 : DIARIZATION (pyannote.audio)")
            print(f"{'─'*60}")
            
            pipeline = load_diarization_pipeline(hf_token)
            if pipeline:
                diarization_segments = diarize_audio(
                    pipeline, audio_path, num_speakers=num_speakers
                )
                
                # Combiner transcription + diarization
                if diarization_segments and transcription["segments"]:
                    dialogue = combine_transcription_diarization(
                        transcription["segments"],
                        diarization_segments
                    )
        else:
            print(f"\n⏭️ ÉTAPE 2 : Diarization ignorée (pas de token HF)")
            print(f"   → Pour activer : $env:HF_TOKEN='ton_token'")
    
    except ImportError:
        print(f"\n⏭️ ÉTAPE 2 : Diarization ignorée (pyannote non installé)")
    
    # =============================================
    # ÉTAPE 3 : VOICE EMBEDDINGS - Optionnel
    # =============================================
    try:
        from src.modules.voice_embeddings import init_voice_encoder, create_voice_embedding
        
        print(f"\n{'─'*60}")
        print(f"📌 ÉTAPE 3 : EMPREINTE VOCALE (Resemblyzer)")
        print(f"{'─'*60}")
        
        encoder = init_voice_encoder()
        if encoder:
            embedding = create_voice_embedding(encoder, audio_path)
            if embedding is not None:
                print(f"   ✅ Empreinte vocale créée ({embedding.shape[0]} dimensions)")
    
    except ImportError:
        print(f"\n⏭️ ÉTAPE 3 : Empreinte vocale ignorée (resemblyzer non installé)")
    
    # =============================================
    # ÉTAPE 4 : ANALYSE DU TEXTE (NLP)
    # =============================================
    print(f"\n{'─'*60}")
    print(f"📌 ÉTAPE 4 : ANALYSE DU TEXTE (NLP)")
    print(f"{'─'*60}")
    
    from src.modules.text_analysis import generate_medical_report
    
    report = generate_medical_report(
        transcription["text"],
        dialogue=dialogue
    )
    
    # =============================================
    # SAUVEGARDE DU RAPPORT
    # =============================================
    save_report(audio_path, transcription, report, dialogue)
    
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
    print(f"{'='*60}")


def save_report(audio_path, transcription, report, dialogue=None):
    """
    Sauvegarde le rapport en format JSON.
    """
    os.makedirs(RAPPORTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    rapport_data = {
        "timestamp": timestamp,
        "audio_file": audio_path,
        "transcription": {
            "text": transcription["text"],
            "language": transcription["language"],
        },
        "analysis": {
            "langue_detectee": report["langue"],
            "symptomes": report["symptomes"],
            "urgence": report["urgence"],
            "hypotheses": report["hypotheses"],
            "questions": report["questions"],
        }
    }
    
    if dialogue:
        rapport_data["dialogue"] = dialogue
    
    json_path = os.path.join(RAPPORTS_DIR, f"{base_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rapport_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 Rapport sauvegardé : {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline médical : Audio → Texte → Analyse → Rapport"
    )
    parser.add_argument("--audio", type=str, required=True, 
                        help="Chemin vers le fichier audio")
    parser.add_argument("--lang", type=str, default=None,
                        help="Langue : fr, ar, ou auto")
    parser.add_argument("--speakers", type=int, default=2,
                        help="Nombre de locuteurs (défaut: 2)")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Taille du modèle Whisper (défaut: base)")
    
    args = parser.parse_args()
    run_pipeline(args.audio, args.lang, args.speakers, args.model)


if __name__ == "__main__":
    main()
