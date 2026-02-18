"""
==========================================================
  DEMO SIMPLE - Tester chaque module individuellement
==========================================================

🎯 Objectif : Script facile pour tester chaque composant
              sans avoir besoin de tout le pipeline

Usage :
  python tests/demo_simple.py --test all        # Tester tout
  python tests/demo_simple.py --test whisper    # Tester Speech-to-Text
  python tests/demo_simple.py --test diarize    # Tester Diarization
  python tests/demo_simple.py --test embedding  # Tester Voice Embeddings
  python tests/demo_simple.py --test nlp        # Tester Text Analysis
  python tests/demo_simple.py --test generate   # Générer les audios de test
==========================================================
"""

import sys
import os
import argparse

# Ajouter le dossier racine au PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import AUDIO_DIR


def test_imports():
    """Vérifie que toutes les bibliothèques sont installées."""
    print("="*60)
    print("  VÉRIFICATION DES IMPORTS")
    print("="*60)
    
    modules = {
        "torch": "PyTorch (Deep Learning framework)",
        "torchaudio": "TorchAudio (traitement audio)",
        "whisper": "OpenAI Whisper (Speech-to-Text)",
        "numpy": "NumPy (calcul mathématique)",
        "scipy": "SciPy (calcul scientifique)",
        "soundfile": "SoundFile (lecture fichiers audio)",
        "librosa": "Librosa (analyse audio)",
        "langdetect": "LangDetect (détection de langue)",
        "gtts": "gTTS (Text-to-Speech pour les tests)",
    }
    
    optional_modules = {
        "pyannote.audio": "pyannote.audio (Speaker Diarization)",
        "resemblyzer": "Resemblyzer (Voice Embeddings)",
        "transformers": "Transformers (Hugging Face NLP)",
    }
    
    print("\n📦 Modules OBLIGATOIRES :")
    all_ok = True
    for module, description in modules.items():
        try:
            __import__(module)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} → pip install {module}")
            all_ok = False
    
    print("\n📦 Modules OPTIONNELS :")
    for module, description in optional_modules.items():
        try:
            __import__(module)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ⚠️ {description} → pip install {module}")
    
    try:
        import torch
        print(f"\n🔧 PyTorch Info :")
        print(f"   Version : {torch.__version__}")
        print(f"   CUDA (GPU) : {'✅ Disponible' if torch.cuda.is_available() else '❌ Non (CPU mode)'}")
        if torch.cuda.is_available():
            print(f"   GPU : {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    
    return all_ok


def test_whisper():
    """Teste le module Speech-to-Text (Whisper)."""
    print("\n" + "="*60)
    print("  TEST : WHISPER (Speech-to-Text)")
    print("="*60)
    
    audio_file = os.path.join(AUDIO_DIR, "test_francais.mp3")
    if not os.path.exists(audio_file):
        print(f"⚠️ Fichier test non trouvé : {audio_file}")
        print("   Génération des audios de test d'abord...")
        test_generate()
    
    from src.modules.speech_to_text import load_model, transcribe_audio
    
    model = load_model("base")
    
    # Test français
    fr_file = os.path.join(AUDIO_DIR, "test_francais.mp3")
    if os.path.exists(fr_file):
        print("\n--- Test Français ---")
        result = transcribe_audio(model, fr_file, language="fr")
    
    # Test arabe
    ar_file = os.path.join(AUDIO_DIR, "test_arabe.mp3")
    if os.path.exists(ar_file):
        print("\n--- Test Arabe ---")
        result = transcribe_audio(model, ar_file, language="ar")
    
    # Test dialogue
    dial_file = os.path.join(AUDIO_DIR, "test_dialogue.mp3")
    if os.path.exists(dial_file):
        print("\n--- Test Dialogue ---")
        result = transcribe_audio(model, dial_file)
    
    print("\n✅ Test Whisper terminé !")


def test_diarization():
    """Teste le module Speaker Diarization (pyannote)."""
    print("\n" + "="*60)
    print("  TEST : PYANNOTE (Speaker Diarization)")
    print("="*60)
    
    try:
        from src.modules.speaker_diarization import load_diarization_pipeline, diarize_audio
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("\n⚠️ Pour tester pyannote, il faut un token Hugging Face :")
            print("   1. Crée un compte sur https://huggingface.co")
            print("   2. Va dans Settings → Access Tokens → New Token")
            print("   3. Accepte les conditions sur :")
            print("      https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   4. Lance : $env:HF_TOKEN='ton_token' (PowerShell)")
            print("   5. Puis relance ce test")
            return
        
        pipeline = load_diarization_pipeline(hf_token)
        if pipeline:
            audio_file = os.path.join(AUDIO_DIR, "test_dialogue.mp3")
            if os.path.exists(audio_file):
                segments = diarize_audio(pipeline, audio_file, num_speakers=2)
            else:
                print(f"⚠️ Fichier non trouvé : {audio_file}")
        
    except ImportError:
        print("❌ pyannote.audio pas installé")
        print("   pip install pyannote.audio")
    
    print("\n✅ Test Diarization terminé !")


def test_embeddings():
    """Teste le module Voice Embeddings (Resemblyzer)."""
    print("\n" + "="*60)
    print("  TEST : RESEMBLYZER (Voice Embeddings)")
    print("="*60)
    
    try:
        from src.modules.voice_embeddings import (
            init_voice_encoder, create_voice_embedding, 
            compare_voices, save_voice_profile
        )
        
        encoder = init_voice_encoder()
        if encoder:
            embeddings = {}
            
            test_files = {
                "francais": os.path.join(AUDIO_DIR, "test_francais.mp3"),
                "dialogue_patient": os.path.join(AUDIO_DIR, "dialogue_patient_fr.mp3"),
                "dialogue_docteur": os.path.join(AUDIO_DIR, "dialogue_docteur_fr.mp3"),
            }
            
            for name, filepath in test_files.items():
                if os.path.exists(filepath):
                    emb = create_voice_embedding(encoder, filepath)
                    if emb is not None:
                        embeddings[name] = emb
                        save_voice_profile(emb, name)
            
            # Comparer les voix
            if len(embeddings) >= 2:
                keys = list(embeddings.keys())
                for i in range(len(keys)):
                    for j in range(i+1, len(keys)):
                        sim = compare_voices(embeddings[keys[i]], embeddings[keys[j]])
                        print(f"\n📊 {keys[i]} vs {keys[j]} → Similarité: {sim:.2%}")
            
            if not embeddings:
                print("⚠️ Aucun fichier audio trouvé")
                print("   Lance : python tests/demo_simple.py --test generate")
    
    except ImportError:
        print("❌ resemblyzer pas installé")
        print("   pip install resemblyzer")
    
    print("\n✅ Test Embeddings terminé !")


def test_nlp():
    """Teste le module Text Analysis (NLP)."""
    print("\n" + "="*60)
    print("  TEST : NLP (Analyse de Texte)")
    print("="*60)
    
    from src.modules.text_analysis import generate_medical_report
    
    tests = [
        ("Français", "J'ai de la fièvre depuis deux jours et je tousse beaucoup. J'ai aussi mal à la gorge et je me sens très fatigué."),
        ("Darija (arabe)", "عندي سخانة و كحة و كنتنفس بصعوبة و راسي كيوجعني"),
        ("Darija (latin)", "ana 3yan bzzaf, 3ndi s5ana w ko7a, rasi kayouj3ni w t3ebt"),
        ("Dialogue", None),
    ]
    
    for name, text in tests:
        if name == "Dialogue":
            dialogue = [
                {"speaker": "PATIENT", "text": "J'ai de la fièvre depuis deux jours et je tousse beaucoup."},
                {"speaker": "DOCTEUR", "text": "Avez-vous des douleurs musculaires ou des maux de tête ?"},
                {"speaker": "PATIENT", "text": "Oui, j'ai mal à la gorge et je me sens très fatigué."},
            ]
            text = " ".join(d["text"] for d in dialogue)
            print(f"\n📝 Test : {name}")
            generate_medical_report(text, dialogue=dialogue)
        else:
            print(f"\n📝 Test : {name}")
            generate_medical_report(text)
    
    print("\n✅ Test NLP terminé !")


def test_generate():
    """Génère les fichiers audio de test."""
    from src.utils.generate_test_audio import generate_all_test_audio
    generate_all_test_audio()


def test_pipeline():
    """Teste le pipeline complet : Audio → Texte → Analyse."""
    print("\n" + "="*60)
    print("  TEST : PIPELINE COMPLET")
    print("="*60)
    print("  Audio → Whisper → Texte → Analyse → Rapport")
    print("="*60)
    
    audio_file = os.path.join(AUDIO_DIR, "test_dialogue.mp3")
    if not os.path.exists(audio_file):
        print("⚠️ Fichier audio manquant, génération...")
        test_generate()
    
    # Étape 1 : Transcrire
    print("\n📌 ÉTAPE 1 : Transcription (Whisper)")
    from src.modules.speech_to_text import load_model, transcribe_audio
    model = load_model("base")
    result = transcribe_audio(model, audio_file)
    
    if result:
        # Étape 2 : Analyser
        print("\n📌 ÉTAPE 2 : Analyse du texte (NLP)")
        from src.modules.text_analysis import generate_medical_report
        report = generate_medical_report(result["text"])
        
        print("\n✅ Pipeline complet terminé !")
    else:
        print("❌ La transcription a échoué")


def main():
    parser = argparse.ArgumentParser(
        description="Demo PFE - Tester les modules individuellement",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--test",
        choices=["all", "imports", "whisper", "diarize", "embedding", "nlp", "generate", "pipeline"],
        default="imports",
        help="""Quel test exécuter :
  imports   → Vérifier les installations (défaut)
  generate  → Générer les audios de test
  whisper   → Tester Speech-to-Text
  diarize   → Tester Speaker Diarization
  embedding → Tester Voice Embeddings
  nlp       → Tester Text Analysis
  pipeline  → Tester le pipeline complet
  all       → Tout tester"""
    )
    
    args = parser.parse_args()
    
    if args.test == "imports":
        test_imports()
    elif args.test == "generate":
        test_generate()
    elif args.test == "whisper":
        test_whisper()
    elif args.test == "diarize":
        test_diarization()
    elif args.test == "embedding":
        test_embeddings()
    elif args.test == "nlp":
        test_nlp()
    elif args.test == "pipeline":
        test_pipeline()
    elif args.test == "all":
        ok = test_imports()
        if ok:
            test_generate()
            test_nlp()
            test_whisper()
            test_embeddings()
            test_pipeline()
        else:
            print("\n❌ Installe d'abord les modules manquants !")


if __name__ == "__main__":
    main()
