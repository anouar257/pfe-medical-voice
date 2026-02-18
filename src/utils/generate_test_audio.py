"""
==========================================================
  GÉNÉRATEUR D'AUDIO DE TEST
==========================================================

🎯 Objectif : Créer des fichiers audio de test pour tester
              tous les modules sans avoir besoin d'un micro

📚 Technologie : gTTS (Google Text-to-Speech)
   → Gratuit, pas besoin de clé API
   → Supporte le Français et l'Arabe

Les fichiers générés seront dans le dossier test_audio/
==========================================================
"""

import os

# Vérifier gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS pas installé : pip install gTTS")


# Dossier de sortie
try:
    from src.config import AUDIO_DIR as OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "audio")


def generate_audio(text, filename, lang="fr"):
    """
    Génère un fichier audio à partir de texte.
    
    Args:
        text: Le texte à convertir en audio
        filename: Nom du fichier de sortie
        lang: Langue ("fr" pour français, "ar" pour arabe)
    """
    if not GTTS_AVAILABLE:
        print("❌ gTTS non disponible")
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    print(f"🔊 Génération : {filename} ({lang})")
    print(f"   Texte : {text[:80]}{'...' if len(text) > 80 else ''}")
    
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filepath)
    
    print(f"   ✅ Sauvegardé : {filepath}")
    return filepath


def generate_all_test_audio():
    """
    Génère tous les fichiers audio de test nécessaires.
    """
    print("="*60)
    print("  GÉNÉRATION DES FICHIERS AUDIO DE TEST")
    print("="*60)
    
    if not GTTS_AVAILABLE:
        print("❌ gTTS non disponible. Installe-le :")
        print("   pip install gTTS")
        return
    
    # ===== 1. Audio en Français (symptômes médicaux) =====
    print("\n📝 1. Audio en Français...")
    generate_audio(
        "J'ai de la fièvre depuis deux jours et je tousse beaucoup. "
        "J'ai aussi mal à la gorge et je me sens très fatigué. "
        "Mon collègue au travail avait la grippe la semaine dernière.",
        "test_francais.mp3",
        lang="fr"
    )
    
    # ===== 2. Audio en Arabe =====
    print("\n📝 2. Audio en Arabe...")
    generate_audio(
        "عندي حمى منذ يومين وأسعل كثيرا. "
        "عندي أيضا ألم في الحلق وأشعر بالتعب الشديد.",
        "test_arabe.mp3",
        lang="ar"
    )
    
    # ===== 3. Dialogue Docteur/Patient (Français) =====
    print("\n📝 3. Dialogue Docteur/Patient (Français)...")
    
    # Partie du Patient
    generate_audio(
        "Bonjour docteur. J'ai de la fièvre depuis deux jours et je tousse beaucoup. "
        "J'ai aussi mal à la gorge.",
        "dialogue_patient_fr.mp3",
        lang="fr"
    )
    
    # Partie du Docteur
    generate_audio(
        "Avez-vous des douleurs musculaires ou des maux de tête ? "
        "Avez-vous été en contact avec quelqu'un de malade récemment ?",
        "dialogue_docteur_fr.mp3",
        lang="fr"
    )
    
    # Dialogue complet (une seule voix pour simplifier)
    generate_audio(
        "Bonjour docteur, j'ai de la fièvre depuis deux jours et je tousse beaucoup. "
        "J'ai aussi mal à la gorge et je me sens très fatigué. "
        "Avez-vous des douleurs musculaires ou des maux de tête ? "
        "Oui, j'ai des maux de tête et des frissons. "
        "Avez-vous été en contact avec quelqu'un de malade ? "
        "Oui, mon collègue avait la grippe.",
        "test_dialogue.mp3",
        lang="fr"
    )
    
    # ===== 4. Audio en Arabe (dialogue médical) =====
    print("\n📝 4. Dialogue médical en Arabe...")
    generate_audio(
        "السلام عليكم يا دكتور، عندي سخونة من يومين وعندي كحة قوية. "
        "وكمان راسي يوجعني وحاسس بتعب شديد.",
        "dialogue_arabe.mp3",
        lang="ar"
    )
    
    print(f"\n{'='*60}")
    print(f"✅ Tous les fichiers audio ont été générés !")
    print(f"📁 Dossier : {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nFichiers créés :")
    
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"   📄 {f} ({size/1024:.1f} KB)")
    
    print(f"\n{'='*60}")
    print(f"🎯 Prochaine étape : tester les modules")
    print(f"   python tests/demo_simple.py --test whisper")
    print(f"   python tests/demo_simple.py --test nlp")
    print(f"   python tests/demo_simple.py --test all")


if __name__ == "__main__":
    generate_all_test_audio()
