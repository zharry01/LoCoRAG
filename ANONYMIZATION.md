# Anonymous Submission Checklist

Use this checklist before sharing the repository link with a paper submission.

## Required

- Create a fresh Git repository instead of pushing this repository's existing history. The current history contains real Git author metadata.
- Push from an anonymous account or use a conference-approved anonymous repository service.
- Do not include `.env`, private API keys, generated logs, result folders, local indexes, or evidence-memory traces.
- Verify there are no author names, emails, institutional identifiers, or local filesystem paths in tracked files.
- Keep the repository private if the venue requires it, and share only the anonymized review link.

## Suggested Fresh-repo Workflow

From this project directory:

```bash
mkdir -p /tmp/locorag-anonymous
rsync -a --delete \
  --exclude .git \
  --exclude .env \
  --exclude __pycache__ \
  --exclude Result \
  --exclude results \
  --exclude outputs \
  --exclude debug \
  --exclude data/evidence_memory \
  --exclude data/log.txt \
  --exclude retriever/indexes \
  ./ /tmp/locorag-anonymous/
cd /tmp/locorag-anonymous
git init
git add .
git -c user.name="Anonymous Authors" -c user.email="anonymous@example.com" commit -m "Initial anonymous submission"
```

Then add the anonymous remote and push from the anonymous account.

## Local Audit Commands

```bash
rg -n "(/U[s]ers/|/h[o]me/|/d[a]ta/[A-Za-z0-9_-]+|@q[q]\\.com|gm[a]il\\.com|edu\\.cn|ac\\.uk|\\.edu)" \
  --glob '!data/**' \
  --glob '!retriever/Corpus/**' \
  .
git log --format='%h %an <%ae> %s'
git status --short
```

Review any matches before publishing.
