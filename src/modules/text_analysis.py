"""
==========================================================
  MODULE 4 : ANALYSE DE TEXTE (NLP)
  Français + Arabe Dialecte Marocain (Darija)
==========================================================

🎯 Objectif : Analyser le texte transcrit pour en extraire des informations médicales
📚 Technologies : langdetect, dictionnaire personnalisé, regex

Fonctionnalités :
1. Détection de langue (Français / Arabe / Darija)
2. Extraction de symptômes médicaux
3. Classification des symptômes par gravité
4. Génération de questions de suivi

⚠️ NOTE SUR LA DARIJA MAROCAINE :
La darija n'est pas une langue officiellement supportée par la plupart
des outils NLP. C'est un dialecte oral de l'arabe, avec :
- Des mots propres au Maroc
- Du mélange français-arabe (code-switching)
- Pas de règles d'orthographe strictes

Notre approche : Dictionnaire personnalisé + détection arabe
==========================================================
"""

import re
import json
from collections import Counter

# Vérifier langdetect
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect pas installé : pip install langdetect")


# ============================================
# DICTIONNAIRES DE SYMPTÔMES
# ============================================

# Symptômes en Français
SYMPTOMES_FR = {
    # Symptômes généraux
    "fièvre": {"gravite": "modéré", "categorie": "général"},
    "température": {"gravite": "modéré", "categorie": "général"},
    "fatigue": {"gravite": "léger", "categorie": "général"},
    "fatigué": {"gravite": "léger", "categorie": "général"},
    "frissons": {"gravite": "modéré", "categorie": "général"},
    "sueurs": {"gravite": "modéré", "categorie": "général"},
    "perte de poids": {"gravite": "modéré", "categorie": "général"},
    "malaise": {"gravite": "modéré", "categorie": "général"},
    
    # Symptômes respiratoires
    "toux": {"gravite": "modéré", "categorie": "respiratoire"},
    "tousser": {"gravite": "modéré", "categorie": "respiratoire"},
    "essoufflement": {"gravite": "sévère", "categorie": "respiratoire"},
    "respiration difficile": {"gravite": "sévère", "categorie": "respiratoire"},
    "mal de gorge": {"gravite": "léger", "categorie": "respiratoire"},
    "gorge": {"gravite": "léger", "categorie": "respiratoire"},
    "nez bouché": {"gravite": "léger", "categorie": "respiratoire"},
    "éternuements": {"gravite": "léger", "categorie": "respiratoire"},
    
    # Douleurs
    "douleur": {"gravite": "modéré", "categorie": "douleur"},
    "mal": {"gravite": "modéré", "categorie": "douleur"},
    "maux de tête": {"gravite": "modéré", "categorie": "neurologique"},
    "migraine": {"gravite": "modéré", "categorie": "neurologique"},
    "douleur thoracique": {"gravite": "sévère", "categorie": "cardiaque"},
    "douleur poitrine": {"gravite": "sévère", "categorie": "cardiaque"},
    "douleur abdominale": {"gravite": "modéré", "categorie": "digestif"},
    "mal au ventre": {"gravite": "modéré", "categorie": "digestif"},
    
    # Symptômes digestifs
    "nausée": {"gravite": "léger", "categorie": "digestif"},
    "vomissement": {"gravite": "modéré", "categorie": "digestif"},
    "diarrhée": {"gravite": "modéré", "categorie": "digestif"},
    "perte d'appétit": {"gravite": "léger", "categorie": "digestif"},
    
    # Symptômes graves
    "perte de conscience": {"gravite": "urgent", "categorie": "neurologique"},
    "convulsions": {"gravite": "urgent", "categorie": "neurologique"},
    "saignement": {"gravite": "sévère", "categorie": "urgent"},
    "sang": {"gravite": "sévère", "categorie": "urgent"},
}

# Symptômes en Darija Marocaine (arabe dialectal)
SYMPTOMES_DARIJA = {
    # Symptômes généraux
    "سخانة": {"fr": "fièvre", "gravite": "modéré", "categorie": "général"},
    "الحمى": {"fr": "fièvre", "gravite": "modéré", "categorie": "général"},
    "عيان": {"fr": "malade/fatigué", "gravite": "léger", "categorie": "général"},
    "عييت": {"fr": "je suis fatigué", "gravite": "léger", "categorie": "général"},
    "مريض": {"fr": "malade", "gravite": "modéré", "categorie": "général"},
    "تعبان": {"fr": "épuisé", "gravite": "modéré", "categorie": "général"},
    
    # Symptômes respiratoires
    "كحة": {"fr": "toux", "gravite": "modéré", "categorie": "respiratoire"},
    "سعال": {"fr": "toux", "gravite": "modéré", "categorie": "respiratoire"},
    "كنكحح": {"fr": "je tousse", "gravite": "modéré", "categorie": "respiratoire"},
    "ضيق التنفس": {"fr": "essoufflement", "gravite": "sévère", "categorie": "respiratoire"},
    "كنتنفس بصعوبة": {"fr": "respiration difficile", "gravite": "sévère", "categorie": "respiratoire"},
    "حلقي كيوجعني": {"fr": "mal de gorge", "gravite": "léger", "categorie": "respiratoire"},
    
    # Douleurs
    "وجع": {"fr": "douleur", "gravite": "modéré", "categorie": "douleur"},
    "كيوجعني": {"fr": "ça me fait mal", "gravite": "modéré", "categorie": "douleur"},
    "راسي كيوجعني": {"fr": "maux de tête", "gravite": "modéré", "categorie": "neurologique"},
    "صداع": {"fr": "maux de tête", "gravite": "modéré", "categorie": "neurologique"},
    "كرشي كيوجعني": {"fr": "mal au ventre", "gravite": "modéré", "categorie": "digestif"},
    "صدري كيوجعني": {"fr": "douleur thoracique", "gravite": "sévère", "categorie": "cardiaque"},
    
    # Symptômes digestifs
    "غادي نتقيأ": {"fr": "envie de vomir", "gravite": "modéré", "categorie": "digestif"},
    "تقيأ": {"fr": "vomissement", "gravite": "modéré", "categorie": "digestif"},
    "إسهال": {"fr": "diarrhée", "gravite": "modéré", "categorie": "digestif"},
    "ما بغيتش ناكل": {"fr": "perte d'appétit", "gravite": "léger", "categorie": "digestif"},
    
    # Symptômes en Darija phonétique (écrit en latin)
    "s5ana": {"fr": "fièvre", "gravite": "modéré", "categorie": "général"},
    "3yan": {"fr": "malade", "gravite": "léger", "categorie": "général"},
    "ko7a": {"fr": "toux", "gravite": "modéré", "categorie": "respiratoire"},
    "rasi kayouj3ni": {"fr": "maux de tête", "gravite": "modéré", "categorie": "neurologique"},
    "karchi kayouj3ni": {"fr": "mal au ventre", "gravite": "modéré", "categorie": "digestif"},
    "sedri kayouj3ni": {"fr": "douleur thoracique", "gravite": "sévère", "categorie": "cardiaque"},
    "t3ebt": {"fr": "je suis fatigué", "gravite": "léger", "categorie": "général"},
}

# Questions de suivi médical
QUESTIONS_SUIVI_FR = [
    "Depuis combien de temps avez-vous ces symptômes ?",
    "La douleur est-elle constante ou intermittente ?",
    "Avez-vous pris des médicaments ?",
    "Avez-vous de la fièvre ? Si oui, quelle température ?",
    "Avez-vous des allergies connues ?",
    "Avez-vous des antécédents médicaux ?",
    "Les symptômes s'aggravent-ils ?",
    "Avez-vous été en contact avec une personne malade ?",
]

QUESTIONS_SUIVI_DARIJA = [
    "شحال هادي عندك هاد الأعراض ؟ (Depuis combien de temps ?)",
    "الوجع دائم ولا كيمشي ويرجع ؟ (Constant ou intermittent ?)",
    "خديتي شي دوا ؟ (Avez-vous pris un médicament ?)",
    "عندك السخانة ؟ شحال ؟ (Avez-vous de la fièvre ?)",
    "عندك شي حساسية ؟ (Avez-vous des allergies ?)",
]


# ============================================
# FONCTIONS D'ANALYSE
# ============================================

def detect_language(text):
    """
    Détecte la langue du texte.
    
    Returns:
        dict avec : language, confidence, is_darija
    """
    result = {
        "language": "unknown",
        "language_name": "Inconnu",
        "confidence": 0.0,
        "is_darija": False
    }
    
    if not text or len(text.strip()) < 3:
        return result
    
    # Vérifier si c'est de la darija (phonétique latin)
    darija_patterns = [
        r'\b(ch7al|3la|dyal|wach|kif|bghit|kayn|mashi|fin|3lach)\b',
        r'\b(kayouj3ni|s5ana|3yan|ko7a|t3ebt)\b',
        r'\b(ana|nta|nti|hna|huma)\b',
    ]
    darija_score = 0
    for pattern in darija_patterns:
        if re.search(pattern, text.lower()):
            darija_score += 1
    
    if darija_score >= 2:
        result["language"] = "darija"
        result["language_name"] = "Darija Marocaine"
        result["confidence"] = 0.8
        result["is_darija"] = True
        return result
    
    # Vérifier si c'est de l'arabe (caractères arabes)
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'\S', text))
    
    if total_chars > 0 and arabic_chars / total_chars > 0.5:
        # Vérifier si c'est de la darija arabe
        for symptome in SYMPTOMES_DARIJA:
            if symptome in text:
                result["is_darija"] = True
                break
        
        result["language"] = "darija" if result["is_darija"] else "ar"
        result["language_name"] = "Darija Marocaine" if result["is_darija"] else "Arabe"
        result["confidence"] = 0.85
        return result
    
    # Utiliser langdetect pour les langues standard
    if LANGDETECT_AVAILABLE:
        try:
            langs = detect_langs(text)
            if langs:
                lang = langs[0]
                result["language"] = str(lang.lang)
                result["confidence"] = lang.prob
                
                lang_names = {
                    "fr": "Français", "ar": "Arabe", "en": "Anglais",
                    "es": "Espagnol", "de": "Allemand"
                }
                result["language_name"] = lang_names.get(str(lang.lang), str(lang.lang))
        except Exception:
            pass
    
    return result


def extract_symptoms(text, language="fr"):
    """
    Extrait les symptômes médicaux du texte.
    
    Args:
        text: Le texte à analyser
        language: "fr" pour français, "darija" pour darija, "ar" pour arabe
    
    Returns:
        Liste de symptômes détectés avec leur gravité
    """
    text_lower = text.lower()
    found_symptoms = []
    
    # Chercher dans le dictionnaire français
    if language in ["fr", "auto"]:
        for symptome, info in SYMPTOMES_FR.items():
            if symptome in text_lower:
                found_symptoms.append({
                    "symptome": symptome,
                    "gravite": info["gravite"],
                    "categorie": info["categorie"],
                    "langue_source": "Français"
                })
    
    # Chercher dans le dictionnaire darija
    if language in ["darija", "ar", "auto"]:
        for symptome, info in SYMPTOMES_DARIJA.items():
            if symptome in text or symptome in text_lower:
                found_symptoms.append({
                    "symptome": info.get("fr", symptome),
                    "symptome_original": symptome,
                    "gravite": info["gravite"],
                    "categorie": info["categorie"],
                    "langue_source": "Darija"
                })
    
    return found_symptoms


def classify_urgency(symptoms):
    """
    Classifie le niveau d'urgence basé sur les symptômes.
    
    Returns:
        "urgent", "sévère", "modéré", ou "léger"
    """
    if not symptoms:
        return "inconnu"
    
    gravites = [s["gravite"] for s in symptoms]
    
    if "urgent" in gravites:
        return "urgent"
    elif "sévère" in gravites:
        return "sévère"
    elif "modéré" in gravites:
        return "modéré"
    else:
        return "léger"


def generate_hypotheses(symptoms):
    """
    Génère des hypothèses diagnostiques basées sur les symptômes.
    (Version simplifiée - sera remplacée par des API médicales)
    """
    if not symptoms:
        return ["Aucun symptôme détecté pour formuler une hypothèse"]
    
    categories = Counter(s["categorie"] for s in symptoms)
    symptom_names = [s["symptome"] for s in symptoms]
    hypotheses = []
    
    # Règles simples basées sur les combinaisons de symptômes
    if "fièvre" in symptom_names and "toux" in symptom_names:
        hypotheses.append("Grippe (Influenza)")
        hypotheses.append("COVID-19")
        if "mal de gorge" in symptom_names:
            hypotheses.append("Angine")
    
    if "fièvre" in symptom_names and "fatigue" in symptom_names:
        hypotheses.append("Infection virale")
    
    if "douleur thoracique" in symptom_names or "douleur poitrine" in symptom_names:
        hypotheses.append("⚠️ Urgence cardiaque possible - Consultation immédiate")
    
    if "maux de tête" in symptom_names:
        hypotheses.append("Céphalée de tension")
        if "fièvre" in symptom_names:
            hypotheses.append("Infection avec composante neurologique")
    
    if "mal au ventre" in symptom_names:
        hypotheses.append("Gastro-entérite")
        if "diarrhée" in symptom_names:
            hypotheses.append("Infection intestinale")
    
    if not hypotheses:
        hypotheses.append("Symptômes non spécifiques - examen complémentaire recommandé")
    
    return hypotheses


def generate_follow_up_questions(symptoms, language="fr"):
    """
    Génère des questions de suivi adaptées.
    """
    questions = []
    
    if language == "darija":
        base_questions = QUESTIONS_SUIVI_DARIJA
    else:
        base_questions = QUESTIONS_SUIVI_FR
    
    # Toujours poser les 3 premières questions de base
    questions.extend(base_questions[:3])
    
    # Ajouter des questions spécifiques selon les symptômes
    categories = set(s["categorie"] for s in symptoms)
    
    if "respiratoire" in categories:
        questions.append("Avez-vous des difficultés à respirer ?" if language == "fr" 
                        else "عندك صعوبة فالتنفس ؟")
    
    if "cardiaque" in categories:
        questions.append("La douleur irradie-t-elle vers le bras gauche ?" if language == "fr"
                        else "الوجع كيمشي للدراع اليسر ؟")
    
    if "digestif" in categories:
        questions.append("Qu'avez-vous mangé récemment ?" if language == "fr"
                        else "شنو كليتي مؤخراً ؟")
    
    return questions


def generate_medical_report(text, dialogue=None):
    """
    Génère un rapport médical structuré en 3 parties
    (comme demandé par le professeur).
    
    Returns:
        dict avec les 3 sections du rapport
    """
    print(f"\n{'='*60}")
    print(f"📋 GÉNÉRATION DU RAPPORT MÉDICAL")
    print(f"{'='*60}")
    
    # 1. Détecter la langue
    lang_info = detect_language(text)
    print(f"\n🌍 Langue détectée : {lang_info['language_name']} ({lang_info['confidence']:.0%})")
    
    # 2. Extraire les symptômes
    lang = "darija" if lang_info["is_darija"] else lang_info["language"]
    symptoms = extract_symptoms(text, language=lang)
    symptoms.extend(extract_symptoms(text, language="auto"))
    
    # Dédupliquer
    seen = set()
    unique_symptoms = []
    for s in symptoms:
        key = s["symptome"]
        if key not in seen:
            seen.add(key)
            unique_symptoms.append(s)
    symptoms = unique_symptoms
    
    # 3. Classifier l'urgence
    urgency = classify_urgency(symptoms)
    
    # 4. Générer les hypothèses
    hypotheses = generate_hypotheses(symptoms)
    
    # 5. Générer les questions
    questions = generate_follow_up_questions(symptoms, language=lang)
    
    # ===== AFFICHAGE DU RAPPORT =====
    
    # PARTIE 1 : Dialogue
    print(f"\n{'─'*60}")
    print(f"1️⃣  DIALOGUE EN TEMPS RÉEL")
    print(f"{'─'*60}")
    if dialogue:
        for entry in dialogue:
            print(f"   🗣️ {entry['speaker']} : {entry['text']}")
    else:
        print(f"   📝 Texte brut : {text}")
    
    # PARTIE 2 : Résultats du rapport
    print(f"\n{'─'*60}")
    print(f"2️⃣  RÉSULTATS DU RAPPORT")
    print(f"{'─'*60}")
    
    print(f"\n   🔎 Symptômes détectés ({len(symptoms)}) :")
    for s in symptoms:
        emoji = {"léger": "🟢", "modéré": "🟡", "sévère": "🟠", "urgent": "🔴"}.get(s["gravite"], "⚪")
        print(f"      {emoji} {s['symptome']} [{s['categorie']}] - Gravité: {s['gravite']}")
    
    urgency_emoji = {"léger": "🟢", "modéré": "🟡", "sévère": "🟠", "urgent": "🔴"}.get(urgency, "⚪")
    print(f"\n   {urgency_emoji} Niveau d'urgence : {urgency.upper()}")
    
    print(f"\n   📋 Hypothèses possibles :")
    for i, h in enumerate(hypotheses, 1):
        print(f"      {i}. {h}")
    
    # PARTIE 3 : Questions de suivi
    print(f"\n{'─'*60}")
    print(f"3️⃣  QUESTIONS PROPOSÉES PAR L'IA")
    print(f"{'─'*60}")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
    
    print(f"\n{'='*60}")
    print(f"⚠️  Ce rapport ne remplace PAS un diagnostic médical réel.")
    print(f"{'='*60}")
    
    return {
        "langue": lang_info,
        "symptomes": symptoms,
        "urgence": urgency,
        "hypotheses": hypotheses,
        "questions": questions
    }


# ===== TEST DIRECT =====
if __name__ == "__main__":
    print("="*60)
    print("  TEST DU MODULE ANALYSE DE TEXTE (NLP)")
    print("="*60)
    
    # Test 1 : Texte en Français
    print("\n\n📝 TEST 1 : Texte en Français")
    print("-"*40)
    texte_fr = "J'ai de la fièvre depuis deux jours et je tousse beaucoup. J'ai aussi mal à la gorge et je me sens très fatigué."
    rapport_fr = generate_medical_report(texte_fr)
    
    # Test 2 : Texte en Darija (arabe)
    print("\n\n📝 TEST 2 : Texte en Darija (arabe)")
    print("-"*40)
    texte_darija = "عندي سخانة و كحة و كنتنفس بصعوبة و راسي كيوجعني"
    rapport_darija = generate_medical_report(texte_darija)
    
    # Test 3 : Texte en Darija (phonétique latin)
    print("\n\n📝 TEST 3 : Texte en Darija (phonétique)")
    print("-"*40)
    texte_darija_latin = "ana 3yan bzzaf, 3ndi s5ana w ko7a, rasi kayouj3ni w t3ebt"
    rapport_darija_latin = generate_medical_report(texte_darija_latin)
    
    # Test 4 : Dialogue simulé
    print("\n\n📝 TEST 4 : Dialogue Docteur/Patient")
    print("-"*40)
    dialogue = [
        {"speaker": "PATIENT", "text": "J'ai de la fièvre depuis deux jours et je tousse beaucoup."},
        {"speaker": "DOCTEUR", "text": "Avez-vous des douleurs musculaires ou des maux de tête ?"},
        {"speaker": "PATIENT", "text": "Oui, j'ai mal à la gorge et je me sens très fatigué."},
        {"speaker": "DOCTEUR", "text": "Avez-vous été en contact avec quelqu'un de malade ?"},
        {"speaker": "PATIENT", "text": "Oui, mon collègue avait la grippe."},
    ]
    texte_complet = " ".join(d["text"] for d in dialogue)
    rapport_dialogue = generate_medical_report(texte_complet, dialogue=dialogue)
    
    print("\n\n✅ Tous les tests terminés !")
