# API Integration README (Framework-Agnostic)

Ce document explique comment consommer l'API CV Parsing depuis n'importe quel framework ou langage.

Technologies internes de parsing:
- OCR: `mistral-ocr-latest`
- LLM parser: `claude-4.5-haiku` (via Replicate)

## 1) Endpoints disponibles

- `GET /health`: etat du service
- `POST /v1/parse-cv`: mode synchrone (one-shot)
- `POST /v1/jobs`: mode asynchrone (creation d'un job)
- `GET /v1/jobs/{job_id}`: suivi et resultat du job
- `GET /docs`: Swagger UI
- `GET /openapi.json`: specification OpenAPI

## 2) Base URL

Exemples:
- Local: `http://localhost:8080`
- Cloud Run: `https://YOUR_SERVICE_URL`

Toutes les routes ci-dessus sont relatives a cette base URL.

## 3) Authentification

L'API supporte un Bearer token:
- Header: `Authorization: Bearer <API_AUTH_TOKEN>`

Comportement:
- Si `API_AUTH_TOKEN` est configure cote serveur: token obligatoire
- Si `API_AUTH_TOKEN` n'est pas configure: endpoint accessible sans token

## 4) Formats de fichiers acceptes

- Extensions: `pdf`, `png`, `jpg`, `jpeg`, `docx`
- Taille max par defaut: `10 MB` (variable `MAX_FILE_SIZE_MB`)
- Upload: `multipart/form-data`, champ fichier nomme `file`

## 5) Choix d'usage: sync vs async

### Mode synchrone: `POST /v1/parse-cv`

Utiliser quand:
- UI interactive simple
- fichiers petits/moyens
- besoin immediat de reponse

Retourne directement le JSON final (ou une erreur).

### Mode asynchrone: `POST /v1/jobs` + `GET /v1/jobs/{job_id}`

Utiliser quand:
- traitement potentiellement long
- besoin de polling/progression
- integration robuste en production

Flux:
1. `POST /v1/jobs` retourne `job_id`
2. Appeler `GET /v1/jobs/{job_id}` jusqu'a `succeeded` ou `failed`

## 6) Contrat HTTP minimal

### 6.1 Health check

`GET /health`

Exemple de reponse:
```json
{
  "status": "ok",
  "time": "2026-02-17T03:00:00Z",
  "request_id": "req_abc123"
}
```

### 6.2 One-shot (sync)

`POST /v1/parse-cv` (`multipart/form-data`)

Champs:
- `file` (required)
- `language_hint` (optional: `fr`, `en`, `ar`, ...)

Reponse succes (`200`):
```json
{
  "status": "succeeded",
  "created_at": "2026-02-17T03:00:00Z",
  "finished_at": "2026-02-17T03:00:08Z",
  "timings": {
    "total_seconds": 8.2,
    "ocr_seconds": 4.1,
    "parsing_seconds": 3.9
  },
  "models": {
    "ocr": "mistral-ocr-latest",
    "llm_parser": "claude-4.5-haiku"
  },
  "result": {},
  "quality": {
    "json_valid": true,
    "schema_valid": true
  },
  "request_id": "req_abc123"
}
```

### 6.3 Async create

`POST /v1/jobs` (`multipart/form-data`)

Champs:
- `file` (required)
- `language_hint` (optional)
- `callback_url` (optional)

Reponse (`202`):
```json
{
  "job_id": "job_01abc...",
  "status": "queued",
  "created_at": "2026-02-17T03:00:00Z",
  "estimated_sla_seconds": 45,
  "request_id": "req_abc123"
}
```

### 6.4 Async status/result

`GET /v1/jobs/{job_id}`

Reponse en cours (`200`):
```json
{
  "job_id": "job_01abc...",
  "status": "processing",
  "created_at": "2026-02-17T03:00:00Z",
  "started_at": "2026-02-17T03:00:01Z",
  "progress": {
    "stage": "ocr",
    "percent": 35
  },
  "request_id": "req_abc123"
}
```

Reponse finale succes (`200`): meme structure que sync, plus `job_id`.

Reponse finale echec (`200`):
```json
{
  "job_id": "job_01abc...",
  "status": "failed",
  "finished_at": "2026-02-17T03:00:09Z",
  "error": {
    "code": "PARSING_SCHEMA_VALIDATION_FAILED",
    "message": "Schema validation failed",
    "details": {}
  },
  "request_id": "req_abc123"
}
```

## 7) Modele d'erreur

Pour les erreurs HTTP (401/404/413/415/422/500...), format:

```json
{
  "error": {
    "code": "STRING_CODE",
    "message": "Human readable message",
    "details": {}
  },
  "request_id": "req_abc123"
}
```

Codes frequents:
- `UNAUTHORIZED`
- `JOB_NOT_FOUND`
- `INVALID_FILE_TYPE`
- `FILE_TOO_LARGE`
- `OCR_PROVIDER_ERROR`
- `LLM_PROVIDER_ERROR`
- `PARSING_JSON_INVALID`
- `PARSING_SCHEMA_VALIDATION_FAILED`
- `INTERNAL_ERROR`

## 8) Exemples d'integration

### 8.1 cURL (sync)

```bash
curl -X POST "$BASE_URL/v1/parse-cv" \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -F "file=@./cv.pdf"
```

### 8.2 cURL (async)

```bash
# Create job
curl -X POST "$BASE_URL/v1/jobs" \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -F "file=@./cv.pdf"

# Poll
curl "$BASE_URL/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $API_AUTH_TOKEN"
```

### 8.3 JavaScript (browser/Node)

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

const createResp = await fetch(`${BASE_URL}/v1/jobs`, {
  method: "POST",
  headers: { Authorization: `Bearer ${API_AUTH_TOKEN}` },
  body: form
});
const job = await createResp.json();

let status = "queued";
while (status === "queued" || status === "processing") {
  await new Promise((r) => setTimeout(r, 2000));
  const r = await fetch(`${BASE_URL}/v1/jobs/${job.job_id}`, {
    headers: { Authorization: `Bearer ${API_AUTH_TOKEN}` }
  });
  const body = await r.json();
  status = body.status;
  if (status === "succeeded") console.log(body.result);
  if (status === "failed") console.error(body.error);
}
```

### 8.4 Python (requests)

```python
import time
import requests

base_url = "https://YOUR_SERVICE_URL"
token = "YOUR_API_AUTH_TOKEN"
headers = {"Authorization": f"Bearer {token}"}

with open("cv.pdf", "rb") as f:
    resp = requests.post(
        f"{base_url}/v1/jobs",
        headers=headers,
        files={"file": ("cv.pdf", f, "application/pdf")},
        timeout=60,
    )
resp.raise_for_status()
job_id = resp.json()["job_id"]

while True:
    r = requests.get(f"{base_url}/v1/jobs/{job_id}", headers=headers, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body["status"] in {"succeeded", "failed"}:
        print(body)
        break
    time.sleep(2)
```

### 8.5 Java (OkHttp)

```java
OkHttpClient client = new OkHttpClient();
RequestBody fileBody = RequestBody.create(new File("cv.pdf"), MediaType.parse("application/pdf"));
RequestBody formBody = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", "cv.pdf", fileBody)
    .build();

Request req = new Request.Builder()
    .url(BASE_URL + "/v1/parse-cv")
    .addHeader("Authorization", "Bearer " + API_AUTH_TOKEN)
    .post(formBody)
    .build();

Response resp = client.newCall(req).execute();
String json = resp.body().string();
```

### 8.6 C# (.NET HttpClient)

```csharp
using var client = new HttpClient();
client.DefaultRequestHeaders.Authorization =
    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiAuthToken);

using var form = new MultipartFormDataContent();
using var stream = File.OpenRead("cv.pdf");
form.Add(new StreamContent(stream), "file", "cv.pdf");

var response = await client.PostAsync($"{baseUrl}/v1/parse-cv", form);
response.EnsureSuccessStatusCode();
var json = await response.Content.ReadAsStringAsync();
```

## 9) Utilisation avec n'importe quel framework

Peu importe votre stack (React, Angular, Vue, Django, Spring Boot, Laravel, .NET, etc.), il faut seulement:
- faire un `multipart/form-data` avec un champ `file`
- envoyer le header Bearer si active
- parser la reponse JSON
- pour l'async: implementer un polling jusqu'a etat terminal

## 10) OpenAPI et generation automatique de client

Vous pouvez generer un SDK client depuis `openapi.json`:

```bash
curl "$BASE_URL/openapi.json" -o openapi.json
npx @openapitools/openapi-generator-cli generate \
  -i openapi.json \
  -g typescript-fetch \
  -o ./generated/ts-client
```

Autres generateurs possibles: `python`, `java`, `csharp`, etc.

## 11) Bonnes pratiques de production

- Ne jamais exposer les cles fournisseur (`MISTRAL_API_KEY`, `REPLICATE_API_TOKEN`) cote frontend.
- Utiliser un token d'API interne (`API_AUTH_TOKEN`) ou un gateway IAM/OAuth.
- Logger `request_id` pour tracer les incidents.
- Preferer le mode async pour les fichiers volumineux.

## 12) Limitation actuelle importante

Le stockage des jobs async est en memoire du process (in-memory).

Consequences:
- un redemarrage du service supprime l'historique des jobs
- pas adapte au scale horizontal strict sans backend partage

Pour une prod robuste, migrer le store jobs vers Redis, Firestore ou SQL.
