# Dear Gentle — Backend

## Aperçu
Backend FastAPI qui alimente "Dear Gentle", une expérience de conversation et d'écriture romanesque centrée sur la persona d'Henry. L'API orchestre les appels OpenAI, applique des garde-fous stylistiques, gère les préférences utilisateurs et structure la mémoire à court et long terme pour conserver la cohérence narrative.

## Pile technique
- **Python 3.11+**
- **FastAPI** pour l'API HTTP
- **Uvicorn** comme serveur ASGI
- **Pydantic v2** pour la validation
- **Structlog** pour la journalisation structurée
- **NumPy** et embeddings OpenAI pour la recherche de contexte

Les dépendances exactes sont listées dans [`requirements.txt`](requirements.txt).

## Démarrage rapide
1. Créez et activez un environnement virtuel Python.
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Définissez les variables d'environnement requises :
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_BASE_URL="https://api.openai.com/v1"  # optionnel
   export OPENAI_CHAT_MODEL="gpt-4o"                  # optionnel
   export OPENAI_EMB_MODEL="text-embedding-3-small"   # optionnel
   export FRONT_ORIGIN="http://localhost:3000"        # optionnel
   ```
4. Lancez le serveur :
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Variables d'environnement
- `OPENAI_API_KEY` *(obligatoire)* : clé API utilisée pour les appels chat & embeddings. L'application lève une exception si elle est absente.
- `OPENAI_BASE_URL` *(défaut : `https://api.openai.com/v1`)*
- `OPENAI_CHAT_MODEL` *(défaut : `gpt‑4o`)*
- `OPENAI_EMB_MODEL` *(défaut : `text-embedding-3-small`)*
- `FRONT_ORIGIN` *(défaut : `http://localhost:3000`)*

## Structure fonctionnelle
- `app.py` : point d'entrée FastAPI, logique principale de construction du contexte, intégration OpenAI et gestion du cycle de conversation/chapitres.
- `models.py` : modèles Pydantic pour messages, snapshots, préférences, livres et chapitres.
- `stores.py` : "bases" en mémoire (utilisateurs, instructions, chapitres, embeddings...).
- `style_packs.py` : définitions des packs de style, contraintes et templates système.
- `auto_memory.py` : capture automatique de faits utilisateur et dédoublonnage sémantique.
- `routes/` : routers spécialisés (styles, instructions, chapitres, health).
- `utils.py` : helpers de temps (Europe/Paris), cosine similarity, entêtes OpenAI.

## API REST (principales routes)
| Méthode & route | Description |
| --- | --- |
| `POST /api/chat` | Conduit un tour de conversation ou génère un chapitre, selon le mode détecté. |
| `POST /api/seed/preferences` | Ajoute ou supprime des préférences utilisateur persistantes. |
| `GET /api/preferences` | Retourne les préférences enregistrées. |
| `POST /api/seed/memories` | Pré-charge des souvenirs long terme (embedding + stockage). |
| `POST /api/snapshot/set` | Définit le snapshot spatio-temporel courant pour la session. |
| `POST /api/book/upsert` | Crée ou met à jour un livre (id client). |
| `GET /api/styles` / `POST /api/styles/select` | Liste les styles disponibles et sélectionne le style actif d'un utilisateur. |
| `GET /api/instructions` | Liste les overrides d'instructions persistantes (commande `>>`). |
| `POST /api/instructions/toggle` / `DELETE /api/instructions` | Active/désactive ou supprime des overrides. |
| `GET /api/chapters` / `GET /api/chapters/{id}` | Liste ou récupère les chapitres avec versioning. |
| `POST /api/chapters/{id}/edit` | Réécrit un chapitre via prompts guidés. |
| `GET /api/health` | Vérification simple de disponibilité. |

## Gestion de la mémoire & du style
- **Mémoire courte** : reconstruction du contexte conversationnel et résumés pour limiter la fenêtre de tokens tout en préservant la cohérence. Les messages sont stockés par session et condensés avant envoi au modèle.
- **Mémoire longue** : souvenirs semés manuellement ou capturés automatiquement, vectorisés pour récupération par similarité cosinus.
- **Auto-mémoire** : `auto_memory.py` extrait jusqu'à 3 faits par message, applique des seuils de confiance (≥0,80 pour auto-validation), et évite les doublons par embeddings.
- **Packs de style** : `style_packs.py` expose des templates système (conversation courte, scène, auteur, rails de sortie) et des contraintes (quota d'émojis, fins interdites). Un endpoint permet de sélectionner le style par utilisateur.

## Modes de conversation
Le backend détecte automatiquement trois modes :
- **Conversation** : réponses courtes ou scénarisées selon le registre détecté (bref / scène).
- **Author** : génère et persiste un nouveau chapitre romanesque, en s'appuyant sur l'outline, les chapitres précédents compressés et la mémoire longue.
- **Instruction (`>>`)** : enregistre des overrides persistants sans appeler l'API OpenAI.

## Persistance & limites actuelles
Toutes les données (utilisateurs, conversations, snapshots, livres, chapitres, mémoires) sont stockées en mémoire. Pour un déploiement production, remplacer ces dictionnaires par des services persistants (base de données, cache distribué, stockage vecteur).

## Développement
- Logging structuré via `structlog` (format JSON + timestamp ISO) dès le démarrage.
- Pas de migrations : la structure est entièrement en mémoire.
- Tests automatiques non fournis pour l'instant — utilisez des appels HTTP manuels ou des tests personnalisés selon vos besoins.

Bon développement !
