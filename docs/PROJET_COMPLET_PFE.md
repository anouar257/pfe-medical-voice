# 🏥 PROJET PFE — Système de Reconnaissance Vocale Médicale

> **📌 Ce fichier est la référence complète du projet. Consultez-le à chaque session de travail.**

---

## 📋 Équipe & Encadrement

- **Projet de 3 personnes** (stage PFE)
- **Professeur encadrant** : supervision et validation
- **Objectif** : Créer un système intelligent d'aide au diagnostic médical basé sur la voix

---

## 🎯 Les 6 Points du Professeur (Cahier des Charges)

### ✅ Point 1 — Connaissance vocale via ordinateur/tablette/phone
- Capter l'audio depuis n'importe quel appareil
- Formats supportés : WAV, MP3, FLAC, OGG
- **Techno utilisée** : `torchaudio` + `soundfile` + `librosa`
- **Fichier** : Gestion audio dans `speech_to_text.py`
- **Statut** : ✅ FAIT ET TESTÉ

### ✅ Point 2 — Vocale → Texte (Speech-to-Text)
- Convertir l'audio en texte transcrit
- Multi-langue : Français + Arabe
- **Techno utilisée** : OpenAI Whisper (modèle `base`)
- **Fichier** : `speech_to_text.py`
- **Statut** : ✅ FAIT ET TESTÉ
- **Résultat** : Transcription correcte FR et AR

### ✅ Point 3 — Différencier les locuteurs via l'empreinte vocale
- Identifier QUI parle QUAND dans un enregistrement
- Créer des empreintes vocales uniques par personne
- **Technos utilisées** :
  - `pyannote.audio` 3.1.1 → Speaker Diarization (state of the art)
  - `Resemblyzer` → Voice Embeddings (empreintes vocales)
- **Fichiers** : `speaker_diarization.py` + `voice_embeddings.py`
- **Statut** : ✅ FAIT ET TESTÉ
- **Résultat** : SPEAKER_00 et SPEAKER_01 séparés, similarité vocale 97.82%

### ✅ Point 4 — Analyse du texte : Français / Arabe dialecte (Darija marocaine)
- Détecter les symptômes médicaux dans le texte
- Supporter le Français ET la Darija marocaine (arabe + phonétique latine)
- Générer un rapport en 3 parties
- **Techno utilisée** : NLP custom avec dictionnaires médicaux
- **Fichier** : `text_analysis.py`
- **Statut** : ✅ FAIT ET TESTÉ
- **Résultat** : 6 symptômes détectés, rapport 3 parties généré

### 🔜 Point 5 — Intégration des API : Symptômes → Diagnostic

#### 5a. Utilisation des API existantes
| API | URL | Type |
|-----|-----|------|
| Infermedica | https://developer.infermedica.com/ | Payant (essai gratuit) |
| ApiMedic | https://apimedic.com/ | Freemium |
| Isabel Healthcare | https://info.isabelhealthcare.com/symptom-checker-api | Payant |

**Logique** : Fièvre + Toux + Fatigue → Grippe probable / Douleur thoracique + Essoufflement → Urgence

**À faire** :
- [ ] Créer un compte sur Infermedica (essai gratuit)
- [ ] Tester l'API : envoyer symptômes → recevoir diagnostics
- [ ] Intégrer dans le pipeline Python
- [ ] Exploiter et extraire les données → stocker dans notre BD

#### 5b. Notre propre API inspirée des API existantes
- [ ] Analyser le format de données des API existantes
- [ ] Créer notre propre base de données de symptômes/diagnostics
- [ ] Développer notre API REST (Spring Boot)
- [ ] Connecter avec le frontend (Angular + Flutter)

### 🔜 Point 6 — Résumé et Conclusion
- [ ] Rapport final du projet
- [ ] Démonstration fonctionnelle
- [ ] Documentation technique complète

---

## 🛠️ Outils et Technologies

### Stack Principale

| Couche | Technologie | Rôle |
|--------|-------------|------|
| **Backend** | Spring Boot (Java) | API REST, logique métier |
| **Frontend Web** | Angular | Interface web |
| **Frontend Mobile** | Flutter | Application mobile |
| **IA / ML** | Python + PyTorch | Traitement audio & NLP |

### Bibliothèques Python (Installées et Testées)

| Package | Version | Rôle |
|---------|---------|------|
| `openai-whisper` | latest | Speech-to-Text (Audio → Texte) |
| `pyannote.audio` | **3.1.1** | Speaker Diarization (Qui parle quand) |
| `resemblyzer` | latest | Voice Embeddings (Empreintes vocales) |
| `torch` | **2.2.0** | Framework Deep Learning |
| `torchaudio` | **2.2.0** | Traitement audio |
| `numpy` | **1.26.4** | Calculs numériques |
| `huggingface_hub` | **0.23.5** | Téléchargement modèles IA |
| `gTTS` | latest | Génération audio de test |
| `librosa` | latest | Analyse audio |
| `langdetect` | latest | Détection de langue |
| `soundfile` | latest | Lecture/écriture audio |

> ⚠️ **IMPORTANT** : Les versions sont critiques ! Ne pas mettre à jour sans tester.

### Outils IA Spécialisés

| Outil | Type | Détails |
|-------|------|---------|
| **OpenAI Whisper** | Speech-to-Text | Multi-langue, gestion accents/bruit |
| **Google Speech-to-Text** | Speech-to-Text (cloud) | Payant, alternative à Whisper |
| **pyannote.audio** | Speaker Diarization | State of the art, réseaux neuronaux |
| **Resemblyzer** | Voice Embeddings | Empreintes vocales, comparaison |
| **NVIDIA NeMo** | Alternative | Framework NVIDIA (non utilisé ici) |
| **PyTorch** | Framework IA | Réseaux neuronaux, modèles pré-entraînés |

### Base de Données Voix (Vector DB)
- Une voix → transformée en **vecteur mathématique** (embedding)
- Ce vecteur est **stocké** dans une base
- On **compare** les nouveaux vecteurs pour identifier la personne
- **Implémentation actuelle** : fichiers `.npy` dans `data/voice_db/`
- **Futur** : Migration vers une vraie Vector DB (Pinecone, Weaviate, ChromaDB)

---

## 📄 Structure d'une Page (Format du Rapport Médical)

### Version : Consultation Docteur / Patient

#### 1️⃣ Dialogue en temps réel (Docteur 🩺 / Patient)

> **Objectif** : Retranscrire fidèlement l'échange sans analyse.

- Conversation naturelle captée par le micro
- Symptômes décrits par le patient
- Questions du médecin
- Aucune interprétation à ce stade
- **Technos** : Whisper (transcription) + pyannote (qui parle)

**Exemple :**
```
Patient : J'ai de la fièvre depuis deux jours et je tousse beaucoup.
Docteur : Avez-vous des douleurs musculaires ou des maux de gorge ?
Patient : Oui, j'ai mal à la gorge et je me sens très fatigué.
Docteur : Avez-vous été en contact avec quelqu'un de malade récemment ?
Patient : Oui, mon collègue avait la grippe.
```

#### 2️⃣ Résultats du rapport

> **Objectif** : Analyser le dialogue et proposer des hypothèses.

**🔎 Analyse des symptômes** : fièvre, toux, fatigue, mal de gorge
**📋 Hypothèses possibles** : Grippe, COVID-19, Rhume
**💡 Propositions médicales** :
- Test de dépistage si suspicion COVID
- Repos + hydratation
- Antipyrétique en cas de forte fièvre
- Surveillance 48h
- Consultation urgente si aggravation

> ⚠️ Ce rapport ne remplace PAS un diagnostic médical réel.

#### 3️⃣ Questions proposées par l'IA

> **Objectif** : Aider à affiner le choix ou la décision.

- Depuis combien de jours les symptômes ont-ils commencé ?
- La fièvre dépasse-t-elle 38,5°C ?
- Avez-vous des difficultés à respirer ?
- Êtes-vous vacciné récemment contre la grippe ou le COVID ?
- Souhaitez-vous consulter immédiatement ou surveiller 24h ?

```
Résumé : 1. Dialogue réel → 2. Analyse + hypothèses → 3. Questions IA
```

---

## 🔬 Extensions Futures (IA pour Laboratoire)

| Domaine | Description |
|---------|-------------|
| Analyses biologiques | Analyse automatique des résultats de labo |
| Analyses anatomopathologiques | Analyse des résultats anapathologiques |
| Imagerie médicale | Analyse des images RX, Scanner, IRM |

---

## 📁 Structure du Projet Python

```
TESTPFE/
├── main.py                        # 🎯 Pipeline complet (point d'entrée)
├── requirements.txt               # Dépendances Python
│
├── src/                           # 📦 Code source
│   ├── __init__.py
│   ├── config.py                  # ⚙️ Configuration des chemins
│   ├── modules/                   # 🧠 Modules IA
│   │   ├── __init__.py
│   │   ├── speech_to_text.py      # Module 1 : Whisper (Audio → Texte)
│   │   ├── speaker_diarization.py # Module 2 : pyannote (Qui parle quand)
│   │   ├── voice_embeddings.py    # Module 3 : Resemblyzer (Empreintes)
│   │   └── text_analysis.py       # Module 4 : NLP (Symptômes FR + Darija)
│   └── utils/                     # 🔧 Utilitaires
│       ├── __init__.py
│       └── generate_test_audio.py # Génération d'audios de test (gTTS)
│
├── tests/                         # 🧪 Tests
│   ├── __init__.py
│   ├── demo_simple.py             # Tests individuels par module
│   └── debug_pyannote.py          # Debug Hugging Face / pyannote
│
├── data/                          # 💾 Données
│   ├── audio/                     # Fichiers audio de test
│   │   ├── test_francais.mp3
│   │   ├── test_arabe.mp3
│   │   ├── test_dialogue.mp3
│   │   ├── dialogue_patient_fr.mp3
│   │   ├── dialogue_docteur_fr.mp3
│   │   └── dialogue_arabe.mp3
│   └── voice_db/                  # Base de données empreintes vocales
│       ├── francais.npy
│       ├── dialogue_patient.npy
│       └── dialogue_docteur.npy
│
├── output/                        # 📊 Sorties
│   └── rapports/                  # Rapports JSON générés
│
└── docs/                          # 📖 Documentation
    ├── GUIDE_TECHNOLOGIES.md      # Guide détaillé des technologies
    └── PROJET_COMPLET_PFE.md      # 📌 CE FICHIER (référence complète)
```

---

## ⚙️ Configuration Technique

### Python
- **Version utilisée** : Python **3.10** (C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe)
- ⚠️ Python 3.13 aussi installé mais les packages sont sur 3.10

### Commandes pour lancer

```powershell
# 1. Rafraîchir le PATH (pour ffmpeg)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 2. Définir le token Hugging Face
$env:HF_TOKEN='VOTRE_TOKEN_HUGGINGFACE'

# 3. Tester tout
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" tests/demo_simple.py --test all

# 4. Tester individuellement
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" tests/demo_simple.py --test whisper
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" tests/demo_simple.py --test diarize
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" tests/demo_simple.py --test nlp
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" tests/demo_simple.py --test pipeline

# 5. Pipeline complet
& "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe" main.py --audio data/audio/test_dialogue.mp3
```

### Hugging Face (pour pyannote)
- **Compte** : anouar10
- **Token** : VOTRE_TOKEN_HUGGINGFACE
- **Modèles acceptés** :
  - ✅ pyannote/speaker-diarization-3.1
  - ✅ pyannote/segmentation-3.0
  - ✅ pyannote/speaker-diarization-community-1

---

## 📊 Avancement Global

| Phase | Points | Statut | Date |
|-------|--------|--------|------|
| Sprint 1 (2 jours) | Points 1-4 | ✅ Terminé | 17 Fév 2026 |
| Sprint 2 | Point 5a (API externes) | 🔜 À faire | — |
| Sprint 3 | Point 5b (Notre API) | 🔜 À faire | — |
| Sprint 4 | Point 6 (Résumé/Conclusion) | 🔜 À faire | — |
| Intégration | Spring Boot + Angular + Flutter | 🔜 À faire | — |
