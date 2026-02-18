# 🏥 PFE — Reconnaissance Vocale Médicale

> **Projet de Fin d'Études** — Système intelligent d'analyse de consultations médicales par reconnaissance vocale.

## 🎯 Objectif

Transformer un enregistrement audio d'une consultation médicale en un **rapport structuré** contenant :
- La **transcription** complète (Français, Arabe, Darija)
- L'**identification** de qui parle (Docteur vs Patient)
- L'**analyse médicale** : symptômes détectés, niveau d'urgence, hypothèses

---

## 🛠️ Prérequis

Avant de commencer, assurez-vous d'avoir :

| Outil | Version | Lien |
|-------|---------|------|
| **Python** | 3.10.x | [python.org](https://www.python.org/downloads/release/python-31011/) |
| **ffmpeg** | Dernière | [ffmpeg.org](https://ffmpeg.org/download.html) |
| **Git** | Dernière | [git-scm.com](https://git-scm.com/) |

> ⚠️ **IMPORTANT** : Python 3.10 est **obligatoire** (pas 3.11, pas 3.12). Les bibliothèques IA ne sont pas compatibles avec les versions plus récentes.

### Token Hugging Face (obligatoire pour la diarization)

1. Créer un compte sur [huggingface.co](https://huggingface.co)
2. Aller dans **Settings → Access Tokens → New Token**
3. Accepter les conditions sur ces 2 modèles :
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

---

## 🚀 Installation (Étape par Étape)

### 1. Cloner le projet
```bash
git clone https://github.com/anouar257/pfe-medical-voice.git
cd pfe-medical-voice
```

### 2. Créer l'environnement virtuel
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```
> ⏱️ La première installation peut prendre **5-10 minutes** (PyTorch fait ~2 Go).

### 4. Configurer le token Hugging Face
```powershell
# Windows (PowerShell)
$env:HF_TOKEN='VOTRE_TOKEN_ICI'
```
```bash
# Linux / macOS
export HF_TOKEN='VOTRE_TOKEN_ICI'
```

### 5. Corriger le PATH pour ffmpeg (Windows uniquement)
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

---

## 🧪 Tester le Projet

### Étape 1 : Vérifier les installations
```bash
python tests/demo_simple.py --test imports
```
Vous devez voir tous les modules en ✅.

### Étape 2 : Générer les fichiers audio de test
```bash
python tests/demo_simple.py --test generate
```
Cela crée des fichiers `.mp3` dans `data/audio/`.

### Étape 3 : Tester chaque module

```bash
# Tester le Speech-to-Text (Whisper)
python tests/demo_simple.py --test whisper

# Tester la Diarization (pyannote) — nécessite HF_TOKEN
python tests/demo_simple.py --test diarize

# Tester les Empreintes Vocales (Resemblyzer + ChromaDB)
python tests/demo_simple.py --test embedding

# Tester l'Analyse de Texte (NLP)
python tests/demo_simple.py --test nlp

# TOUT tester d'un coup
python tests/demo_simple.py --test all
```

### Étape 4 : Lancer le pipeline complet
```bash
python main.py --audio data/audio/test_dialogue.mp3
```
Cela exécute les 4 étapes : **Audio → Transcription → Diarization → Analyse → Rapport JSON**.

---

## 📁 Structure du Projet

```
TESTPFE/
├── src/
│   ├── config.py                  # Configuration des chemins
│   ├── modules/                   # 🧠 Modules IA
│   │   ├── speech_to_text.py         # Whisper (Audio → Texte)
│   │   ├── speaker_diarization.py    # pyannote (Qui parle quand)
│   │   ├── voice_embeddings.py       # Resemblyzer (Empreintes vocales)
│   │   ├── vector_db_manager.py      # ChromaDB (Vector Database)
│   │   └── text_analysis.py          # NLP (Symptômes FR + Darija)
│   └── utils/
│       └── generate_test_audio.py    # Génération d'audios de test
├── tests/
│   ├── demo_simple.py                # Tests unitaires & globaux
│   └── debug_pyannote.py             # Debug Hugging Face
├── data/
│   └── audio/                        # Fichiers MP3 de test
├── docs/                             # Documentation
├── output/rapports/                  # Rapports JSON générés
├── main.py                           # 🎯 Pipeline principal
├── requirements.txt                  # Dépendances Python
└── .gitignore
```

---

## 📦 Technologies Utilisées

| Technologie | Rôle |
|-------------|------|
| **Python 3.10** | Langage principal |
| **PyTorch** | Framework Deep Learning |
| **OpenAI Whisper** | Speech-to-Text (multi-langue) |
| **pyannote.audio** | Speaker Diarization |
| **Resemblyzer** | Empreintes vocales (embeddings) |
| **ChromaDB** | Vector Database (stockage embeddings) |
| **NLP Custom** | Analyse médicale (FR + Darija) |

---

## ⚠️ Problèmes Courants

<details>
<summary><b>❌ ModuleNotFoundError: No module named 'xxx'</b></summary>

Vérifiez que le virtual environment est activé :
```powershell
.\.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate        # Linux
```
Puis réinstallez : `pip install -r requirements.txt`
</details>

<details>
<summary><b>❌ Token HF_TOKEN non trouvé</b></summary>

```powershell
$env:HF_TOKEN='votre_token'     # Windows PowerShell
export HF_TOKEN='votre_token'   # Linux / macOS
```
</details>

<details>
<summary><b>❌ ffmpeg not found</b></summary>

1. Télécharger depuis [ffmpeg.org](https://ffmpeg.org/download.html)
2. Ajouter au PATH système
3. Relancer PowerShell
</details>

<details>
<summary><b>❌ pyannote 403 Forbidden</b></summary>

Vous n'avez pas accepté les conditions des modèles :
1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Cliquez "Agree and access repository" sur chaque page.
</details>

---

## 👥 Équipe

- **Anouar** — Développeur principal
- Collègue 1
- Collègue 2

---

## 📌 Sprints

- [x] **Sprint 1** : Modules Python IA (Points 1-4) ✅
- [x] **Sprint 1.5** : Vector Database ChromaDB ✅
- [ ] **Sprint 2** : API Externes (Infermedica)
- [ ] **Sprint 3** : Spring Boot + Angular + Flutter
- [ ] **Sprint 4** : Rapport final
