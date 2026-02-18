"""
==========================================================
  MODULE 3b : GESTIONNAIRE VECTOR DB (ChromaDB)
==========================================================

🎯 Objectif : Gérer le stockage et la recherche des embeddings vocaux
📚 Technologie : ChromaDB (Vector Database Open Source)

Fonctionnalités :
1. Initialiser la base de données (persistante sur disque)
2. Ajouter un locuteur (Nom + Vecteur)
3. Identifier un locuteur (Recherche du vecteur le plus proche)
4. Lister tous les locuteurs connus

Avantages par rapport aux fichiers .npy :
- Plus rapide pour la recherche (indexation)
- Scalable (peut gérer des milliers de voix)
- Gestion des métadonnées (ID, nom, dates, etc.)
==========================================================
"""

import os
import chromadb
import numpy as np

# Chemin de la base de données
try:
    from src.config import CHROMA_DB_DIR as DB_PATH
except ImportError:
    # Fallback pour tests isolés
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chroma_db")


class VectorDBManager:
    def __init__(self, collection_name="voice_embeddings"):
        """
        Initialise la connexion à ChromaDB.
        
        Args:
            collection_name: Nom de la collection (table) à utiliser.
        """
        print(f"🗄️  Connexion à ChromaDB dans : {DB_PATH}")
        
        # Client persistant = sauvegarde sur disque
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # Créer ou récupérer la collection
        # distance="cosine" est optimal pour comparer des embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection '{collection_name}' prête (contient {self.collection.count()} voix).")


    def add_speaker(self, name, embedding):
        """
        Ajoute ou met à jour un locuteur dans la base.
        
        Args:
            name: Nom du locuteur (sera utilisé comme ID)
            embedding: Vecteur numpy (256 float)
        """
        # ChromaDB demande des listes Python, pas des numpy arrays
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        print(f"💾 Sauvegarde de {name} dans ChromaDB...")
        
        self.collection.upsert(
            ids=[name],               # ID unique = Nom du speaker
            embeddings=[embedding],   # Le vecteur
            metadatas=[{"name": name}] # Métadonnées optionnelles
        )
        print(f"✅ {name} sauvegardé avec succès.")


    def identify_speaker(self, embedding, threshold=0.75):
        """
        Cherche le locuteur le plus proche de l'embedding donné.
        
        Args:
            embedding: Vecteur de la voix inconnue
            threshold: Seuil de similarité (0 à 1). 
                       Attention : ChromaDB renvoie une "distance" (0=identique, 1=différent).
                       Donc seuil de similarité 0.75 <=> distance < 0.25 (environ).
                       
        Returns:
            Nom du locuteur ou "INCONNU"
        """
        if self.collection.count() == 0:
            return "INCONNU"
            
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        # Rechercher le voisin le plus proche (n_results=1)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1
        )
        
        # Analyser le résultat
        if not results["ids"] or not results["ids"][0]:
            return "INCONNU"
            
        best_match_id = results["ids"][0][0]
        distance = results["distances"][0][0]
        
        # Convertir la distance cosine en similarité (approximatif pour l'affichage)
        # Distance Cosine varie de 0 (identique) à 2 (opposé).
        # On veut savoir si c'est "proche" (distance petite).
        
        # Seuil empirique pour ChromaDB (distance cosine)
        # Si distance < 0.3, c'est très probablement la même personne.
        threshold_distance = 1 - threshold  # ex: 1 - 0.75 = 0.25
        
        print(f"🔍 Résultat ChromaDB : {best_match_id} (Distance: {distance:.4f})")
        
        # Note: Avec cosine distance, plus c'est petit, mieux c'est.
        # On utilise 0.3 comme seuil de sécurité standard.
        if distance < 0.3: 
            return best_match_id
        else:
            print(f"❌ Trop éloigné (Seuil distance < 0.3)")
            return "INCONNU"


    def list_speakers(self):
        """Retourne la liste de tous les locuteurs enregistrés."""
        # get() sans arguments retourne tout (limité par défaut, mais ok pour PFE)
        data = self.collection.get()
        return data["ids"]


# ===== TEST DIRECT =====
if __name__ == "__main__":
    print("="*60)
    print("  TEST DU MODULE VECTOR DB (ChromaDB)")
    print("="*60)
    
    # 1. Init
    db = VectorDBManager()
    
    # 2. Fake Data
    fake_name = "Test_User"
    fake_vector = np.random.rand(256).tolist() # Vecteur aléatoire
    
    # 3. Add
    db.add_speaker(fake_name, fake_vector)
    
    # 4. List
    print(f"📋 Locuteurs : {db.list_speakers()}")
    
    # 5. Search (doit trouver le même)
    print("🔎 Recherche du même vecteur...")
    match = db.identify_speaker(fake_vector)
    print(f"👉 Résultat : {match}")
    
    # 6. Search random (doit échouer)
    print("🔎 Recherche d'un vecteur différent...")
    random_vector = np.random.rand(256).tolist()
    match = db.identify_speaker(random_vector)
    print(f"👉 Résultat : {match}")
