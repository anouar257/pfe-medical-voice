# 🧠 Guide des Technologies - PFE Reconnaissance Vocale Médicale

## Vue d'ensemble

```
Audio (mic/fichier) 
    ↓
[Whisper] Speech-to-Text → Texte transcrit
    ↓
[pyannote] Speaker Diarization → Qui parle quand  
    ↓
[Resemblyzer] Voice Embeddings → Identifier la personne
    ↓
[NLP] Text Analysis → Symptômes, urgence, hypothèses
    ↓
📋 Rapport structuré en 3 parties
```

---

## 1. PyTorch — Le moteur de Deep Learning

**C'est quoi ?** Framework open-source créé par Meta (Facebook) pour le Deep Learning.

**Pourquoi on l'utilise ?** C'est la fondation sur laquelle Whisper, pyannote et Resemblyzer sont construits.

**Concepts clés :**
- **Tensor** = Tableau de nombres (comme un array NumPy, mais peut tourner sur GPU)
- **Modèle** = Réseau de neurones qui a appris à faire une tâche
- **Pré-entraîné** = Modèle déjà entraîné sur des millions de données

```python
import torch
# Un tensor simple
x = torch.tensor([1.0, 2.0, 3.0])
print(x)  # tensor([1., 2., 3.])
print(torch.cuda.is_available())  # False = CPU, True = GPU
```

---

## 2. OpenAI Whisper — Speech-to-Text

**C'est quoi ?** Modèle d'IA créé par OpenAI qui convertit la parole en texte.

**Caractéristiques :**
- ✅ **Gratuit et open-source** (pas comme Google Speech API)
- ✅ **Multi-langue** : 99+ langues (français, arabe, anglais...)
- ✅ **Fonctionne en local** (pas besoin d'internet après le téléchargement)
- ✅ **Gère le bruit** et les accents

**Comment ça marche (simplifié) :**
```
Audio → Mel Spectrogram → Transformer Encoder → Transformer Decoder → Texte
```

1. L'audio est converti en **spectrogramme** (image des fréquences)
2. Un **Encoder** (réseau de neurones) analyse cette image
3. Un **Decoder** (autre réseau) génère le texte mot par mot

**Tailles de modèle :**

| Modèle | Paramètres | Vitesse CPU | Précision |
|--------|-----------|-------------|-----------|
| tiny   | 39M       | ⚡ Très rapide | ⭐⭐ |
| base   | 74M       | ⚡ Rapide    | ⭐⭐⭐ |
| small  | 244M      | 🔄 Moyen     | ⭐⭐⭐⭐ |
| medium | 769M      | 🐢 Lent      | ⭐⭐⭐⭐⭐ |
| large  | 1550M     | 🐌 Très lent | ⭐⭐⭐⭐⭐ |

**Code exemple :**
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3", language="fr")
print(result["text"])
# "J'ai de la fièvre depuis deux jours et je tousse beaucoup."
```

---

## 3. pyannote.audio — Speaker Diarization

**C'est quoi ?** Bibliothèque Python qui répond à la question : **"Qui parle quand ?"**

**Le terme "Diarization" :**
- Le mot vient de "diary" (journal)
- C'est comme écrire un journal de qui parle à chaque instant

**Comment ça marche :**
```
Audio → Détection d'activité vocale → Segmentation → Clustering → Labels
```

1. **VAD** (Voice Activity Detection) : Quand y a-t-il de la parole ?
2. **Segmentation** : Découper l'audio en petits morceaux
3. **Embedding** : Convertir chaque morceau en vecteur
4. **Clustering** : Regrouper les morceaux similaires (même voix)

**Résultat :**
```
[0.0s → 4.5s]  SPEAKER_00  (= Docteur)
[4.5s → 8.2s]  SPEAKER_01  (= Patient)
[8.2s → 12.0s] SPEAKER_00  (= Docteur)
```

**⚠️ Nécessite un token Hugging Face (gratuit)**

---

## 4. Resemblyzer — Empreintes Vocales

**C'est quoi ?** Bibliothèque qui crée une "empreinte digitale" unique pour chaque voix.

**Comment ça marche :**
```
Audio de 5 secondes → Réseau de neurones → Vecteur de 256 nombres
```

Ce vecteur (appelé **embedding**) est unique pour chaque personne :
```python
# Voix de Ahmed  → [0.12, -0.45, 0.78, ...]  (256 nombres)
# Voix de Sara   → [0.89, 0.23, -0.56, ...]  (256 nombres)
# Nouvelle voix  → [0.11, -0.44, 0.77, ...]  → Ressemble à Ahmed !
```

**Comparaison avec similarité cosinus :**
- Score > 0.75 → **Même personne**
- Score < 0.75 → **Personnes différentes**

---

## 5. NLP — Analyse de Texte (Français + Darija)

**C'est quoi ?** NLP = Natural Language Processing (Traitement du Langage Naturel)

**Notre approche pour le projet :**

### Français
- Dictionnaire de symptômes médicaux avec catégories et gravité
- Détection automatique via `langdetect`
- Extraction par correspondance de mots-clés

### Darija Marocaine 🇲🇦
**Défi :** La darija n'est PAS une langue standard dans les outils NLP car :
- C'est un dialecte oral (pas de règles d'orthographe)
- Mélange d'arabe, français et berbère
- Peut s'écrire en arabe OU en latin ("arabizi")

**Notre solution :**
- Dictionnaire personnalisé de symptômes en darija (arabe + latin)
- Détection via patterns regex pour la darija phonétique
- Fallback sur détection arabe standard

**Exemples darija :**
| Darija (arabe) | Darija (latin) | Français |
|----------------|---------------|----------|
| سخانة | s5ana | fièvre |
| كحة | ko7a | toux |
| راسي كيوجعني | rasi kayouj3ni | maux de tête |
| عيان | 3yan | malade |
| كرشي كيوجعني | karchi kayouj3ni | mal au ventre |

---

## 6. Le Rapport en 3 Parties

Le pipeline génère un rapport structuré comme demandé par le professeur :

### Partie 1 : Dialogue en temps réel
```
🗣️ PATIENT : J'ai de la fièvre depuis deux jours...
🗣️ DOCTEUR : Avez-vous des douleurs musculaires ?
🗣️ PATIENT : Oui, j'ai mal à la gorge...
```

### Partie 2 : Résultats du rapport
```
🔎 Symptômes : fièvre (modéré), toux (modéré), fatigue (léger)
⚠️ Urgence : MODÉRÉ
📋 Hypothèses : Grippe, COVID-19, Angine
```

### Partie 3 : Questions de suivi
```
1. Depuis combien de temps avez-vous ces symptômes ?
2. La douleur est-elle constante ou intermittente ?
3. Avez-vous des difficultés à respirer ?
```

---

## Commandes pour tester

```powershell
# Utiliser le bon Python (3.10)
$py = "C:\Users\anoua\AppData\Local\Programs\Python\Python310\python.exe"

# Rafraîchir le PATH (pour ffmpeg)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Tester les imports
& $py demo_simple.py --test imports

# Générer les audios de test
& $py demo_simple.py --test generate

# Tester le NLP (analyse de texte)
& $py demo_simple.py --test nlp

# Tester Whisper (Speech-to-Text)
& $py demo_simple.py --test whisper

# Tester le pipeline complet
& $py demo_simple.py --test pipeline

# Ou directement :
& $py main.py --audio test_audio/test_dialogue.mp3
& $py main.py --audio test_audio/test_arabe.mp3 --lang ar
```
