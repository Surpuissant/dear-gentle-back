# Dear Gentle — Backend

## Aperçu
Backend FastAPI qui alimente "Dear Gentle", un "book boyfriend" virtuel que l'on apprivoise au fil d'échanges très longs. Chaque utilisateur converse avec un gentleman en devenir (Henry par défaut) dont la personnalité et les rails d'écriture proviennent d'un **style pack** dédié. L'API orchestre les appels OpenAI, applique des garde-fous stylistiques, gère la mémoire de la relation et construit des chapitres romanesques à partir des conversations.

### Concept narratif
- **Phase de séduction** : l'utilisateur discute pendant des dizaines de pages (10 à 30 pages équivalent) avec son gentleman. Celui-ci conserve les informations importantes, développe une voix crédible et fait évoluer la relation.
- **Mémoire vivante** : le gentleman capture, pondère et réutilise les faits pertinents (préférences, biographie, engagements, détails de scènes) pour rester cohérent même après de longues sessions.
- **Ancrage spatio-temporel** : les messages prennent en compte les indications de l'utilisateur (heure, météo, ambiance). Le snapshot courant peut être mis à jour dynamiquement pour refléter le soir, la pluie ou tout autre contexte partagé.
- **Chapitres romancés** : à tout moment, l'utilisateur peut convoquer l'"auteur" (un second mode LLM) pour transformer l'échange en chapitre. Les chapitres sont stockés, résumés et réinjectés pour nourrir la suite de l'histoire.

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
- `app.py` : point d'entrée FastAPI, logique principale de construction du contexte, intégration OpenAI et gestion du cycle de conversation/chapitres (gentleman vs auteur).
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
- **Mémoire courte** : reconstruction du contexte conversationnel et résumés rapides pour limiter la fenêtre de tokens tout en préservant la cohérence. Les messages sont stockés par session et reformatés avant envoi au modèle.
- **Mémoire longue** : souvenirs semés manuellement ou capturés automatiquement, vectorisés pour récupération par similarité cosinus. Les chapitres enregistrés alimentent également la continuité du récit.
- **Auto-mémoire** : `auto_memory.py` extrait jusqu'à 3 faits par message, applique des seuils de confiance (≥0,80 pour auto-validation), évite les doublons par embeddings et impose un cooldown pour ne pas répéter les mêmes souvenirs.
- **Packs de style** : `style_packs.py` expose des templates système (conversation courte, scène, auteur, rails de sortie) et des contraintes (quota d'émojis, fins interdites). Un endpoint permet de sélectionner le style par utilisateur et donc d'alterner entre différents gentlemen.
- **Chapitres** : chaque chapitre romancé est résumé et indexé (embeddings) pour être réutilisé lors des prochains tours ou des réécritures.

## Modes de conversation
Le backend détecte automatiquement trois modes :
- **Conversation** : réponses courtes ou scénarisées selon le registre détecté (bref / scène). Le gentleman incarne son style pack tout en tenant compte du snapshot spatio-temporel.
- **Author** : génère et persiste un nouveau chapitre romanesque, en s'appuyant sur l'outline du livre, les chapitres précédents compressés et la mémoire longue (faits utilisateur + chapitres).
- **Instruction (`>>`)** : enregistre des overrides persistants (consignes, contraintes) sans appeler l'API OpenAI.

## Persistance & limites actuelles
Toutes les données (utilisateurs, conversations, snapshots, livres, chapitres, mémoires) sont stockées en mémoire. Pour un déploiement production, remplacer ces dictionnaires par des services persistants (base de données, cache distribué, stockage vecteur).

## Scope POC & pistes d'amélioration
Ce dépôt illustre un POC fonctionnel pour prouver la viabilité du concept "book boyfriend".
- **Ce qui est couvert** : sélection de style pack, gestion des préférences, capture automatique de souvenirs, continuité chapitre par chapitre, orchestrations OpenAI.
- **À garder en tête pour la suite** : persistance durable, modèle de mémoire plus riche (résumés multi-sessions, tags narratifs), multiples gentlemen packagés, détection automatique des indices temporels/météo, interface de validation des auto-mémoires.
- **Objectif** : livrer rapidement une base crédible sur laquelle itérer sans dépasser le scope POC.

## Développement
- Logging structuré via `structlog` (format JSON + timestamp ISO) dès le démarrage.
- Pas de migrations : la structure est entièrement en mémoire.
- Tests automatiques non fournis pour l'instant — utilisez des appels HTTP manuels ou des tests personnalisés selon vos besoins.

Bon développement !
