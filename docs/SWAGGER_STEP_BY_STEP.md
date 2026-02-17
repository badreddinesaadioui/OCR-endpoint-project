# Swagger Step-by-Step (Test rapide API)

Ce guide est fait pour tester l'API sans ambiguite, clic par clic.

## 1) Demarrer l'API

Dans PowerShell:

```powershell
cd "c:\Users\asus\Desktop\forvis mazar\OCR-endpoint-project"
.\.venv\Scripts\uvicorn cv_api.main:app --host 0.0.0.0 --port 8080 --reload
```

Puis ouvre dans le navigateur:
- `http://localhost:8080/docs`

Important:
- `0.0.0.0` = adresse d'ecoute serveur
- dans le navigateur, utilise `localhost`

## 2) Verifier que le service tourne

Dans Swagger:
1. Ouvre `GET /health`
2. Clique `Try it out`
3. Clique `Execute`

Tu dois voir `200` et `"status": "ok"`.

## 3) Test principal en mode async (recommande)

### Etape A - creer un job
1. Ouvre `POST /v1/jobs`
2. Clique `Try it out`
3. Champ `file`: upload un CV, par exemple:
   - `ground_truth_database/cv/cv005.pdf`
4. Laisse `language_hint` vide (optionnel)
5. Laisse `callback_url` vide (optionnel)
6. Clique `Execute`

Reponse attendue:
- status HTTP `202`
- JSON avec:
  - `job_id`
  - `status = queued`
  - `created_at`

Copie la valeur `job_id`.

### Etape B - suivre le job
1. Ouvre `GET /v1/jobs/{job_id}`
2. Clique `Try it out`
3. Colle ton `job_id`
4. Clique `Execute`

Tu verras un de ces statuts:
1. `queued`
2. `processing`
3. `succeeded`
4. `failed`

Si `queued` ou `processing`, relance `Execute` apres 2-5 secondes.

### Etape C - lire le resultat final

Quand `status = succeeded`, la reponse contient:
1. `timings` (temps OCR, parsing, total)
2. `models` (OCR + LLM utilises)
3. `result` (JSON final structure)
4. `quality` (`json_valid`, `schema_valid`)

Si `status = failed`, lis:
- `error.code`
- `error.message`

## 4) Test alternatif en mode sync

1. Ouvre `POST /v1/parse-cv`
2. Clique `Try it out`
3. Upload un fichier CV
4. Clique `Execute`

Tu recois directement le JSON final (pas de `job_id`).

## 5) Si auth Bearer est active

Si `API_AUTH_TOKEN` est defini, Swagger demandera un token:
1. Clique `Authorize` (en haut)
2. Saisis:
   - `Bearer ton-token`
3. Clique `Authorize`

Ensuite refais les appels.

## 6) Erreurs courantes et correction rapide

1. `INVALID_FILE_TYPE`
- Cause: extension non supportee
- Fix: utilise `pdf|png|jpg|jpeg|docx`

2. `FILE_TOO_LARGE`
- Cause: fichier > limite
- Fix: reduire taille ou augmenter `MAX_FILE_SIZE_MB`

3. `OCR_PROVIDER_ERROR`
- Cause: souci cote OCR/API key
- Fix: verifier `MISTRAL_API_KEY`

4. `LLM_PROVIDER_ERROR`
- Cause: souci cote LLM/API key
- Fix: verifier `REPLICATE_API_TOKEN`

5. `PARSING_JSON_INVALID` ou `PARSING_SCHEMA_VALIDATION_FAILED`
- Cause: sortie modele invalide
- Fix: relancer, verifier texte OCR, verifier prompt/timeout

## 7) Ce que represente exactement le lien `/docs`

Exemple: `http://localhost:8080/docs`

1. `http://` = protocole
2. `localhost` = ta machine locale
3. `8080` = port de ton API
4. `/docs` = page Swagger UI (interface de test)

Autres liens:
- `http://localhost:8080/openapi.json` = spec OpenAPI brute
- `http://localhost:8080/redoc` = doc alternative
