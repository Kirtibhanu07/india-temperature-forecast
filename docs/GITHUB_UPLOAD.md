# Publish the prepared repository

Proposed owner/name: `Kirtibhanu07/india-temperature-forecast`.
The requested visibility is **public**. The repository is prepared locally;
creation/upload require an authenticated GitHub session.

1. Sign in to GitHub and open [New repository](https://github.com/new).
2. Use `india-temperature-forecast`, select **Public**, and leave automatic README,
   license and `.gitignore` initialization off because those files are already prepared.
3. Create the empty repository. From this local project folder, add the destination
   and push the prepared `main` branch:

```bash
git remote add origin https://github.com/Kirtibhanu07/india-temperature-forecast.git
git push -u origin main
```

If Git asks for authentication, use GitHub's supported sign-in/credential flow.
Never paste a token into source files, a notebook, or the README.

After pushing, check the README's six figures, the model folder/manifest, and the
**Actions** tab for the first hosted test run. The local check result does not
establish that a hosted run passed.

The accompanying export ZIP contains repository files, excluding `.git`, `.venv`,
raw/processed station downloads, runtime caches and local credentials. Extract it
before working with the project. Uploading the ZIP itself as a single GitHub file
will not render its README, directories or figures as a normal repository.

Choose a software license separately before describing the project as open-source.
The upstream observation-data terms remain separate from any code license.
