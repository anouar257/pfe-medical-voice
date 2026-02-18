"""
==========================================================
  MODULE 2 : SPEAKER DIARIZATION avec pyannote.audio
==========================================================

🎯 Objectif : Identifier QUI parle QUAND dans un audio
📚 Technologie : pyannote.audio (state of the art)

Comment ça marche ?
- pyannote.audio utilise des réseaux neuronaux profonds
- Il analyse l'audio et détecte :
  1. Quand quelqu'un commence à parler
  2. Quand quelqu'un arrête de parler
  3. Quelle voix correspond à quel locuteur
  
- Le résultat ressemble à :
  [0:00 → 0:05] SPEAKER_00 (Docteur)
  [0:05 → 0:12] SPEAKER_01 (Patient)
  [0:12 → 0:18] SPEAKER_00 (Docteur)

⚠️ IMPORTANT : Il faut un token Hugging Face !
  1. Crée un compte sur https://huggingface.co
  2. Va dans Settings → Access Tokens → New Token
  3. Accepte les conditions sur :
     https://huggingface.co/pyannote/speaker-diarization-3.1
     https://huggingface.co/pyannote/segmentation-3.0
==========================================================
"""

import os
import json

# On vérifie si pyannote est installé
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ pyannote.audio n'est pas installé.")
    print("   Installe-le avec : pip install pyannote.audio")


def load_diarization_pipeline(hf_token=None):
    """
    Charge le pipeline de diarization pré-entraîné.
    
    Méthode : On utilise huggingface_hub.login() pour s'authentifier
    globalement, puis on charge le pipeline sans passer le token
    directement (évite les problèmes de compatibilité de versions).
    
    Args:
        hf_token: Token Hugging Face (obligatoire)
    
    Returns:
        Pipeline de diarization
    """
    if not PYANNOTE_AVAILABLE:
        print("❌ pyannote.audio n'est pas disponible")
        return None
    
    if not hf_token:
        # Chercher dans les variables d'environnement
        hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("❌ Token Hugging Face manquant !")
        print("   Utilise : $env:HF_TOKEN='ton_token_ici' (PowerShell)")
        print("   Ou passe le token en paramètre")
        return None
    
    print("📥 Chargement du pipeline de diarization...")
    print("   (Premier chargement = téléchargement du modèle)")
    
    try:
        # Étape 1 : Login global avec le token
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        
        # Étape 2 : Charger le pipeline (le token est déjà enregistré)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
            
        print("✅ Pipeline de diarization chargé !")
        return pipeline
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        print("   Causes possibles :")
        print("   1. Token incorrect (vérifie qu'il est en mode 'Read')")
        print("   2. Conditions non acceptées sur Hugging Face")
        print("      👉 https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("      👉 https://huggingface.co/pyannote/segmentation-3.0")
        print("      👉 https://huggingface.co/pyannote/speaker-diarization-community-1")
        return None


def diarize_audio(pipeline, audio_path, num_speakers=None):
    """
    Effectue la diarization d'un fichier audio.
    
    Args:
        pipeline: Pipeline pyannote chargé
        audio_path: Chemin vers le fichier audio
        num_speakers: Nombre de locuteurs attendus (optionnel)
    
    Returns:
        Liste de segments : [{"start": float, "end": float, "speaker": str}]
    """
    if pipeline is None:
        print("❌ Pipeline non chargé")
        return []
    
    if not os.path.exists(audio_path):
        print(f"❌ Fichier non trouvé : {audio_path}")
        return []
    
    print(f"\n🔍 Diarization de : {audio_path}")
    print(f"   Nombre de locuteurs : {'Auto' if num_speakers is None else num_speakers}")
    
    # Lancer la diarization
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    
    diarization = pipeline(audio_path, **kwargs)
    
    # Extraire les segments
    segments = []
    print(f"\n{'='*60}")
    print(f"📋 RÉSULTAT DE LA DIARIZATION")
    print(f"{'='*60}")
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
            "duration": round(turn.end - turn.start, 2)
        }
        segments.append(segment)
        print(f"   [{turn.start:.1f}s → {turn.end:.1f}s] 🗣️ {speaker}")
    
    # Statistiques
    speakers = set(s["speaker"] for s in segments)
    print(f"\n📊 Statistiques :")
    print(f"   Nombre de locuteurs détectés : {len(speakers)}")
    for spk in speakers:
        total_time = sum(s["duration"] for s in segments if s["speaker"] == spk)
        print(f"   {spk} → {total_time:.1f}s de parole")
    print(f"{'='*60}")
    
    return segments


def combine_transcription_diarization(whisper_segments, diarization_segments):
    """
    Combine la transcription Whisper avec la diarization pyannote
    pour créer un dialogue structuré : QUI dit QUOI.
    
    C'est la fonction clé qui crée le format docteur/patient !
    """
    dialogue = []
    
    for w_seg in whisper_segments:
        w_start = w_seg["start"]
        w_end = w_seg["end"]
        text = w_seg["text"].strip()
        
        # Trouver le locuteur correspondant
        best_speaker = "INCONNU"
        best_overlap = 0
        
        for d_seg in diarization_segments:
            # Calculer le chevauchement temporel
            overlap_start = max(w_start, d_seg["start"])
            overlap_end = min(w_end, d_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]
        
        dialogue.append({
            "speaker": best_speaker,
            "text": text,
            "start": w_start,
            "end": w_end
        })
    
    # Afficher le dialogue
    print(f"\n{'='*60}")
    print(f"💬 DIALOGUE RECONSTITUÉ")
    print(f"{'='*60}")
    for entry in dialogue:
        print(f"   🗣️ {entry['speaker']} : {entry['text']}")
    print(f"{'='*60}")
    
    return dialogue


# ===== TEST DIRECT =====
if __name__ == "__main__":
    import sys
    # Ajouter la racine du projet au PYTHONPATH
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PROJECT_ROOT)
    from src.config import AUDIO_DIR

    print("="*60)
    print("  TEST DU MODULE SPEAKER DIARIZATION (pyannote)")
    print("="*60)
    
    if PYANNOTE_AVAILABLE:
        print(f"\n✅ pyannote.audio importé avec succès !")
        
        # Tester le chargement du pipeline
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            pipeline = load_diarization_pipeline(hf_token)
            if pipeline:
                # Tester avec un fichier audio
                test_file = os.path.join(AUDIO_DIR, "test_dialogue.mp3")
                if os.path.exists(test_file):
                    segments = diarize_audio(pipeline, test_file, num_speakers=2)
                else:
                    print(f"\n⚠️ Fichier test non trouvé : {test_file}")
                    print(f"   → Lance d'abord : python tests/demo_simple.py --test generate")
        else:
            print("\n⚠️ Token Hugging Face non trouvé !")
            print("   Pour tester, fais :")
            print("   $env:HF_TOKEN='ton_token'  (PowerShell)")
            print("   Puis relance ce script")
    else:
        print("\n❌ pyannote.audio pas installé")
        print("   pip install pyannote.audio")

