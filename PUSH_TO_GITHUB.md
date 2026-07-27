# Push CRANIUM to your GitHub

This folder is **already a git repository** with a first commit. You just need to
create a repo on your GitHub account and push to it. Two ways — pick one.

> Why you have to run this yourself: the assistant's sandbox can only reach
> repositories that were pre-connected to its session, so it can't create a new
> repo on your account. These commands run from *your* machine, where you're
> logged in to GitHub. Takes about 30 seconds.

Open a terminal in this folder first:
```bash
cd path/to/cranium-mvp
```

## Option A — GitHub CLI (one command)
If you have the `gh` CLI (https://cli.github.com) and are logged in (`gh auth login`):
```bash
gh repo create cranium-mvp --private --source=. --remote=origin --push
```
That creates a **private** repo named `cranium-mvp` on your account and pushes.
Swap `--private` for `--public` if you want it public.

## Option B — plain git
1. Create an empty repo at https://github.com/new — name it `cranium-mvp`,
   choose Private or Public, and **do not** tick "Add a README / .gitignore /
   license" (this folder already has them).
2. Then:
```bash
git remote add origin https://github.com/<your-username>/cranium-mvp.git
git branch -M main
git push -u origin main
```
If prompted for a password, use a **Personal Access Token** (GitHub no longer
accepts your account password): https://github.com/settings/tokens

## Notes
- The landmark model `models/face_landmarker.task` (~3.7 MB) is committed so the
  repo clones and runs out of the box. It's under GitHub's 50 MB soft limit, so
  plain git is fine. If you'd rather keep binaries out of git, move it to
  Git LFS later (`git lfs track "*.task"`).
- The initial commit is authored as you (Khalik Oketokun). Change it any time
  with `git commit --amend --author="Name <email>"` before pushing.
- After pushing, enable GitHub Pages (Settings → Pages → deploy from `main`,
  `/demo` is not a Pages root by default — either move `cranium_demo.html` to the
  repo root or set the Pages source folder) to host the live demo over HTTPS,
  which is what the webcam needs.
