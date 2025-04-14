# AI Document Question-Answering System

Un système de questions-réponses basé sur des documents PDF, utilisant Ollama, LangChain, et Flask. Le système permet de charger des documents PDF et de poser des questions sur leur contenu, avec une interface web moderne et une API RESTful.

## 📋 Architecture du Système

### Architecture du Modèle
```
                                    ┌─────────────────┐
                                    │     Ollama      │
                                    │  (Mistral LLM)  │
                                    └────────┬────────┘
                                            │
                    ┌─────────────────────┬─┴───┬─────────────────────┐
                    │                     │     │                     │
            ┌───────┴───────┐    ┌───────┴───┐ │             ┌───────┴───────┐
            │  PDF Parser    │    │   FAISS   │ │             │  LangChain    │
            │  (PDFPlumber)  │    │   Index   │ │             │    Chains     │
            └───────┬───────┘    └───────┬───┘ │             └───────┬───────┘
                    │                    │     │                     │
                    └────────────────────┴─────┴─────────────────────┘
                                            │
                                    ┌───────┴───────┐
                                    │  Flask App    │
                                    │    & API      │
                                    └───────────────┘
```

### Flux de Données
1. **Traitement des Documents**
   - Chargement du PDF via PDFPlumber
   - Découpage en chunks de texte (taille configurable)
   - Génération des embeddings via sentence-transformers
   - Stockage dans l'index FAISS

2. **Traitement des Questions**
   - Génération de l'embedding de la question
   - Recherche des chunks pertinents dans FAISS
   - Combinaison avec le prompt via LangChain
   - Génération de réponse via Ollama (Mistral)

### Stockage et Persistence
```
storage/
├── faiss_index/           # Index vectoriel FAISS
│   ├── index.faiss       # Vecteurs des documents
│   └── index.pkl         # Métadonnées de l'index
├── docs_metadata.json    # Information sur les documents
└── uploads/             # Documents PDF temporaires
```

## 👨‍💼 Interface d'Administration

L'interface d'administration est accessible à `/admin` et offre les fonctionnalités suivantes :

### Gestion des Documents
- Vue d'ensemble des documents chargés
- Statistiques d'utilisation
- Suppression de documents
- Réindexation manuelle

### Monitoring du Système
- État du service Ollama
- Utilisation de la mémoire
- Taille de l'index FAISS
- Logs du système

### Configuration du Modèle
- Paramètres de chunking
  ```python
  chunk_size = 1000       # Taille des segments
  chunk_overlap = 200     # Chevauchement
  ```
- Paramètres de recherche
  ```python
  search_kwargs = {
      "k": 3,            # Nombre de chunks à récupérer
      "score_threshold": 0.5  # Seuil de pertinence
  }
  ```
- Configuration Ollama
  ```python
  llm_config = {
      "model": "mistral",
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 500
  }
  ```

### Format des Réponses

Les réponses du modèle suivent cette structure :
```json
{
    "answer": "Réponse générée",
    "sources": [
        {
            "document": "nom_du_document.pdf",
            "page": 1,
            "confidence": 0.92,
            "excerpt": "Extrait pertinent du document"
        }
    ],
    "metadata": {
        "processing_time": "0.8s",
        "chunks_used": 3,
        "model": "mistral"
    }
}
```

### Métriques et Logs

L'interface d'administration affiche :
- Temps de réponse moyen
- Taux de succès des requêtes
- Utilisation des ressources
- Historique des questions/réponses

### Sécurité de l'Admin
- Authentification requise
- Logs des actions administrateur
- Restrictions IP configurables
- Sessions avec timeout

## 🔧 Variables d'Environnement

Créez un fichier `.env` avec :
```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=votre_mot_de_passe_securise
OLLAMA_API_HOST=http://localhost:11434
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS=50
LOG_LEVEL=INFO
```

## 🚀 Prérequis

### Système
- Python 3.8+
- [Ollama](https://ollama.ai/) installé et configuré
- Au moins 8GB de RAM recommandé
- Espace disque suffisant pour stocker les documents et les index

### Installation d'Ollama

1. Installer Ollama depuis [ollama.ai](https://ollama.ai/)
2. Télécharger le modèle Mistral :
```bash
ollama pull mistral
```

3. Vérifier que Ollama fonctionne :
```bash
ollama run mistral "Bonjour!"
```

## 📦 Installation

1. Cloner le repository :
```bash
git clone [votre-repo]
cd [votre-repo]
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 🛠️ Configuration

Le système utilise plusieurs composants clés :
- **Ollama** : Modèle de langage local
- **FAISS** : Index vectoriel pour la recherche sémantique
- **Flask** : Serveur web et API
- **LangChain** : Framework pour les chaînes de traitement LLM

### Structure des dossiers
```
.
├── app.py          # Application web principale
├── api.py          # API RESTful
├── storage/        # Stockage des index et métadonnées
│   ├── faiss_index/
│   └── docs_metadata.json
└── templates/      # Templates HTML
```

## 🚀 Démarrage

### Application Web
```bash
python app.py
```
L'application web sera accessible sur `http://localhost:5000`

### API
```bash
python api.py
```
L'API sera accessible sur `http://localhost:5001`

## 🔌 Utilisation de l'API

### Endpoints disponibles

1. **Statut du système**
```http
GET /api/status
```
Réponse :
```json
{
    "status": "ready",
    "has_documents": true,
    "document_count": 2,
    "docs_metadata": [...]
}
```

2. **Poser une question**
```http
POST /api/ask
Content-Type: application/json

{
    "question": "Votre question ici"
}
```

3. **Liste des documents**
```http
GET /api/documents
```

### Exemple d'utilisation (Python)
```python
import requests

# Poser une question
response = requests.post('http://localhost:5001/api/ask', 
    json={'question': 'Votre question'})
print(response.json())

# Vérifier le statut
status = requests.get('http://localhost:5001/api/status').json()
print(status)
```

## ⚙️ Configuration avancée

### Ollama

Le système utilise Ollama avec le modèle Mistral. Pour modifier le modèle :

1. Télécharger un autre modèle :
```bash
ollama pull llama2
# ou
ollama pull codellama
```

2. Modifier le modèle dans `api.py` :
```python
llm = ChatOllama(model="llama2")  # ou autre modèle
```

### Paramètres de recherche

Dans `api.py`, vous pouvez ajuster les paramètres de recherche :
```python
retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
```
- `k` : nombre de chunks de texte à récupérer (augmenter pour plus de contexte)

## 🔒 Sécurité

- L'API utilise CORS pour la sécurité des requêtes cross-origin
- Les fichiers PDF sont validés avant le traitement
- Limite de taille des fichiers à 10MB

## 🚨 Dépannage

1. **Ollama n'est pas accessible**
   - Vérifier que le service Ollama est en cours d'exécution
   - Par défaut, Ollama écoute sur `localhost:11434`

2. **Erreurs de mémoire**
   - Réduire la taille des chunks de texte
   - Augmenter la RAM disponible
   - Limiter le nombre de documents chargés simultanément

3. **Problèmes de performance**
   - Vérifier la charge CPU/GPU
   - Ajuster les paramètres de chunking
   - Optimiser la taille de l'index FAISS

## 📝 Maintenance

- Les index FAISS sont sauvegardés automatiquement
- Les métadonnées des documents sont conservées dans `storage/docs_metadata.json`
- Nettoyer régulièrement le dossier `storage` si nécessaire

## 🔄 Mise à jour

Pour mettre à jour le système :
1. Arrêter les serveurs
2. Pull les dernières modifications
3. Mettre à jour les dépendances : `pip install -r requirements.txt`
4. Redémarrer les serveurs

## 📄 Licence

[Votre licence ici] 