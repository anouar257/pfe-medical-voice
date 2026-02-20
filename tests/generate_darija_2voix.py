"""
Génère un dialogue médical en DARIJA avec 2 voix différentes.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gtts import gTTS
from pydub import AudioSegment
from src.config import AUDIO_DIR

OUTPUT_FILE = os.path.join(AUDIO_DIR, "dialogue_darija_2voix.mp3")

# Dialogue médical en Darija (arabe marocain)
DIALOGUE = [
    ("docteur", "السلام عليكم، أنا الدكتور. شنو عندك اليوم؟"),
    ("patient", "عليكم السلام يا دكتور. عندي سخونة من تلات ايام وكنسعل بزاف."),
    ("docteur", "واش قستي الحرارة ديالك؟"),
    ("patient", "اه يا دكتور، كانت تسعة وتلاتين. وعندي حتى وجع فالراس وفالحلق."),
    ("docteur", "واش عندك وجع فالجسم ولا تعب؟"),
    ("patient", "اه والله، عندي تعب كبير ومقدرتش نعس. جسمي كولو كيوجعني."),
    ("docteur", "واش كان شي واحد مريض حداك فالخدمة؟"),
    ("patient", "اه دكتور، واحد الزميل ديالي كان عندو لاگريب الاسبوع اللي فات."),
]


def generate():
    print("=" * 60)
    print("  GÉNÉRATION DIALOGUE DARIJA 2 VOIX")
    print("=" * 60)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    combined = AudioSegment.silent(duration=500)

    for i, (speaker, text) in enumerate(DIALOGUE):
        print(f"\n🎤 [{speaker.upper()}] : {text[:50]}...")

        if speaker == "docteur":
            tts = gTTS(text=text, lang="ar", tld="com", slow=False)
        else:
            tts = gTTS(text=text, lang="ar", tld="co.uk", slow=False)

        temp_file = os.path.join(AUDIO_DIR, f"_temp_{i}.mp3")
        tts.save(temp_file)

        segment = AudioSegment.from_mp3(temp_file)

        if speaker == "patient":
            segment = segment._spawn(segment.raw_data, overrides={
                "frame_rate": int(segment.frame_rate * 0.90)
            }).set_frame_rate(segment.frame_rate)

        combined += segment
        combined += AudioSegment.silent(duration=800)

        os.remove(temp_file)

    combined.export(OUTPUT_FILE, format="mp3")
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024

    print(f"\n{'=' * 60}")
    print(f"✅ Dialogue Darija 2 voix généré !")
    print(f"📁 Fichier : {OUTPUT_FILE}")
    print(f"📊 Taille  : {size_kb:.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    generate()
