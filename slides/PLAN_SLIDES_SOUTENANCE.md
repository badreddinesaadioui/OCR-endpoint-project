# Plan des slides – Soutenance (tuteur & consultant)

**Public :** Tuteur (Adil Ahidar) et consultant Forvis Mazars — pas hyper technique, focus sur le contexte, la démarche et les résultats.

**Contexte projet :** Pipeline CV → JSON pour l’ATS Forvis Mazars. Notre partie : **OCR** (extraire le texte du CV) + **parsing LLM** (structurer en JSON). API déployée sur **Google Cloud Run**, consommée par une **app POC** (Svelte).

---

## 1. Ouverture (2–3 slides)

### Slide 1 – Titre
- **Titre :** Pipeline CV → JSON : OCR et parsing
- **Sous-titre :** Projet Option – Groupe 10 – Forvis Mazars
- **Tuteur :** Adil Ahidar
- **Équipe :** LIMI Zakaria, NAHLI Ghita, OUTZOULA Abderrazzak, SALEHI Abderrahmane, SAADIOU Badreddine

### Slide 2 – Contexte métier (simple)
- **Problème :** Forvis Mazars reçoit des CV variés (PDF, Word, images, plusieurs langues, mises en page différentes). Le traitement manuel est long et peu exploitable pour la CVthèque.
- **Objectif du sous-projet :** Transformer un CV (fichier) en **données structurées (JSON)** pour alimenter l’ATS et la CVthèque.
- **Notre périmètre :** deux briques clés : **OCR** (lire le texte dans le document) et **parsing** (extraire nom, expériences, compétences, etc. en JSON).

*Pas de schéma technique compliqué — une phrase du type : « Entrée = fichier CV → Sortie = JSON structuré » suffit.*

### Slide 3 – Ce que nous livrons (vision produit)
- **Une API** déployée sur **Google Cloud Run** : on envoie un CV, on reçoit le JSON.
- **Une application POC** (interface web) pour tester l’API : dépôt de CV, bouton « Parser », affichage du résultat.
- **Un choix de modèles** justifié par des **benchmarks** : quels moteurs OCR et quel moteur de parsing utiliser, et pourquoi.

*Image optionnelle : logo Forvis Mazars si vous en avez un.*

---

## 2. Corpus et démarche d’évaluation (2 slides)

### Slide 4 – La CVthèque de test
- Pour comparer les solutions de manière **objective**, nous avons constitué une **base de 16 CV** variés :
  - **Formats :** PDF, Word, images (png, jpg)
  - **Langues :** français, anglais, arabe
  - **Mises en page :** une colonne, deux colonnes, type « infographique »
  - **Types :** CV numériques ou scannés
- Cette base sert à **tester** les moteurs OCR et de parsing, et à **mesurer** précision, rapidité et coût.

*Image : `01-cvtheque-database.png` (Our CVtheque).*

### Slide 5 – Méthode : benchmark puis décision
- **Étape 1 – Candidats :** Nous avons testé plusieurs moteurs :
  - **OCR :** Mistral OCR 3, OpenAI (vision), Deepseek OCR, Marker, text-extract-ocr (certains via Replicate car pas d’API directe).
  - **Parsing :** GPT 4.1 nano, GPT 5 mini (OpenAI), Gemini 2.5 Flash, Claude 4.5 Haiku (Replicate).
- **Étape 2 – Critères :** précision (erreurs de caractères/mots, respect du layout, qualité du JSON), **temps de réponse**, **coût**.
- **Étape 3 – Décision :** nous avons utilisé des **méthodes de vote multi-critères** (Borda, Condorcet) pour ne pas trancher sur un seul chiffre.

*Pas besoin d’entrer dans les formules — dire que « nous avons comparé plusieurs critères de façon structurée ».*

---

## 3. OCR : candidats et résultats (3 slides)

### Slide 6 – Benchmark OCR (qui, quoi)
- **Modèles testés :** Mistral OCR 3 (Mistral), GPT-4o vision (OpenAI), Deepseek OCR et Marker (Replicate), text-extract-ocr (Replicate).
- **Données :** les 16 CV de la base + texte de référence (« ground truth ») pour calculer les taux d’erreur (CER, WER) et la fidélité de la mise en page.
- **Contrainte opérationnelle :** nous avons exclu les modèles dont le temps de réponse dépassait **20 secondes** (trop lent pour un usage fluide).

*Image : `02-ocr-benchmark-ui.png` + éventuellement `04-logo-mistral.png`, `06-logo-replicate.png` (et OpenAI si vous avez le logo).*

### Slide 7 – Résultats OCR : les deux finalistes
- Après filtrage (vitesse < 20 s), **deux modèles** sont retenus pour la comparaison finale :
  - **Mistral OCR 3**
  - **Replicate text-extract-ocr**
- Tableau récapitulatif : **précision** (CER, WER, layout), **temps moyen**, **coût total** sur la base.

*Image : `08-parallel-ocr-results-verdict.png` (tableau + verdict).*

### Slide 8 – Résultats OCR : graphiques et verdict
- **Graphiques :** précision selon le type de layout (mono, multi, infographique) et selon la langue ; temps moyen par run ; coût.
- **Verdict :** **Mistral OCR 3** offre le meilleur **compromis** (meilleure précision et meilleur layout ; Replicate est plus rapide et moins cher, mais moins précis). Nous le choisissons pour l’API.

*Images : `09-ocr-charts-cer-time.png`, `16-ocr-charts-cer-layout-language.png`, `17-ocr-charts-time-cost.png` (en choisir 1 ou 2 pour ne pas surcharger).*

---

## 4. Parsing LLM : candidats et résultats (3 slides)

### Slide 9 – Benchmark parsing (qui, quoi)
- **Modèles testés :** GPT 4.1 nano, GPT 5 mini (OpenAI), Gemini 2.5 Flash, Claude 4.5 Haiku (Replicate).
- **Entrée :** texte issu de l’OCR (ou texte brut du CV). **Sortie attendue :** JSON structuré (nom, expériences, formations, compétences, etc.).
- **Métriques :** qualité d’extraction (comparaison au « ground truth »), **validité du JSON** et du **schéma**, temps et coût.
- **Contrainte :** nous avons exclu **GPT 5 mini** car trop lent (**plus de 15 secondes** en moyenne).

*Image : `11-llm-parsing-benchmark-ui.png`.*

### Slide 10 – Résultats parsing : tableau et verdict
- **Résumé par modèle :** précision moyenne, temps moyen, coût total, % de JSON valides et conformes au schéma.
- **Verdict :** **Claude 4.5 Haiku** est retenu : **meilleure précision**, JSON et schéma toujours valides, temps et coût raisonnables.

*Image : `12-parallel-llm-parsing-results-verdict.png`.*

### Slide 11 – Résultats parsing : graphiques
- Précision selon le type de layout ; temps selon la langue ; temps moyen par run.
- Renforce le message : Claude 4.5 Haiku est à la fois **précis** et **rapide** sur notre base.

*Image : `13-llm-parsing-charts-accuracy-time.png`.*

---

## 5. Synthèse des choix (1 slide)

### Slide 12 – Recommandations finales
- **OCR :** **Mistral OCR 3** — meilleur compromis précision / layout / vitesse / coût.
- **Parsing :** **Claude 4.5 Haiku** — meilleure précision, JSON et schéma valides, coût maîtrisé.
- Ces deux modèles sont **intégrés dans l’API** que nous déployons.

*Image : `15-results-summary-ocr-llm.png` (résumé OCR + LLM).*

---

## 6. Livrable : API et application (2 slides)

### Slide 13 – L’API (ce qu’on a déployé)
- **Où :** Google Cloud Run (service managé, HTTPS, mise à l’échelle automatique).
- **Fonctionnement simple :** le client envoie un fichier CV (PDF, Word, image) ; l’API appelle **Mistral OCR 3** puis **Claude 4.5 Haiku** ; elle renvoie un **JSON structuré**.
- **Endpoints utiles :** santé du service (`/health`), parsing synchrone (`POST /v1/parse-cv`), et option asynchrone (création de job + suivi) pour les cas plus lourds.
- **Sécurité :** clés des fournisseurs (Mistral, Replicate) dans **Google Secret Manager** ; authentification par token pour l’API si besoin.

*Pas de détail d’implémentation — rester au niveau « service disponible, sécurisé, prêt à être branché à l’ATS ».*

### Slide 14 – L’application POC
- **Rôle :** démontrer l’usage de l’API en conditions réelles : l’utilisateur dépose un CV, clique sur « Parser », et voit le JSON extrait.
- **Techno :** application web (Svelte) qui appelle l’API déployée sur Cloud Run.
- **Intérêt :** preuve que la chaîne **CV → API → JSON** fonctionne de bout en bout, et base possible pour une future interface recruteur.

*Image : `18-cv-parser-svelte-app-ui.png` (interface « forvis mazars » – dépôt de CV, bouton Parse CV).*

---

## 7. Clôture (1–2 slides)

### Slide 15 – Récap et suite
- **Récap :** corpus de 16 CV → benchmarks OCR et parsing → choix **Mistral OCR 3** + **Claude 4.5 Haiku** → **API sur Cloud Run** + **POC Svelte**.
- **Suite possible :** intégration avec l’ATS Forvis, extension des formats, amélioration des modèles ou des seuils (vitesse, coût) selon les retours métier.

### Slide 16 – Merci / Questions
- **Merci** à notre tuteur **Adil Ahidar** et au consultant Forvis Mazars.
- **Questions.**

---

## Récapitulatif des images à utiliser

| Slide  | Fichier image suggéré |
|--------|------------------------|
| 4      | `01-cvtheque-database.png` |
| 6      | `02-ocr-benchmark-ui.png`, logos Mistral/Replicate (04, 06) |
| 7      | `08-parallel-ocr-results-verdict.png` |
| 8      | `09-ocr-charts-cer-time.png` ou 16 / 17 |
| 9      | `11-llm-parsing-benchmark-ui.png` |
| 10     | `12-parallel-llm-parsing-results-verdict.png` |
| 11     | `13-llm-parsing-charts-accuracy-time.png` |
| 12     | `15-results-summary-ocr-llm.png` |
| 14     | `18-cv-parser-svelte-app-ui.png` |

Les tableaux « All runs » (`14-llm-parsing-all-runs-table.png` et éventuellement un équivalent OCR) peuvent être gardés en **annexe** ou en slide de secours si une question détaille les runs.

---

## Ton et conseils

- **Éviter :** formules Borda/Condorcet, détails d’API (codes HTTP, schémas JSON), Docker/Cloud Build.
- **Préférer :** « nous avons comparé plusieurs moteurs », « nous avons fixé une limite de temps », « nous avons choisi le meilleur compromis », « l’API est en ligne et utilisable par l’app POC ».
- **Insister sur :** la **démarche** (corpus → benchmark → critères → décision), la **livraison** (API + POC), et le **lien avec le cahier des charges** (pipeline CV → JSON, multilingue, déployé et prêt pour l’intégration).
