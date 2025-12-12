# 🚀 GitHub Setup Guide

This guide will help you push PlantDocBot to GitHub.

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon → **"New repository"**
3. Fill in the details:
   - **Repository name:** `plantdocbot` (or your preferred name)
   - **Description:** `AI-Powered Plant Disease Detection System using Deep Learning`
   - **Visibility:** Public (or Private)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Link Local Repository to GitHub

Copy the commands from GitHub (they'll look similar to below) and run them:

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/plantdocbot.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

## Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files uploaded
3. Check that README.md displays properly

## Step 4: Upload Model Files (Separately)

Since model files are large (>100MB), upload them separately:

### Option A: Google Drive

1. Upload `Backend/models/` to Google Drive
2. Make the folder shareable
3. Update `Backend/models/.gitkeep` with the download link
4. Commit and push the update:
   ```bash
   git add Backend/models/.gitkeep
   git commit -m "docs: add model download link"
   git push
   ```

### Option B: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pth"
git lfs track "*.safetensors"
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Add and commit models
git add Backend/models/
git commit -m "feat: add trained models via Git LFS"
git push
```

**Note:** Git LFS has storage limits on free tier.

### Option C: HuggingFace Hub

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload models
huggingface-cli upload YOUR_USERNAME/plantdocbot-models ./Backend/models/
```

Then update your code to download models from HuggingFace.

## Step 5: Update README

Update the following in `README.md`:

1. **Line 164:** Replace `yourusername` with your GitHub username
   ```markdown
   git clone https://github.com/YOUR_USERNAME/plantdocbot.git
   ```

2. **Line 475+:** Update contact information and links

3. **Add model download link** in the "Model Files" section

Commit and push:
```bash
git add README.md
git commit -m "docs: update repository links"
git push
```

## Step 6: Add Repository Topics

On GitHub repository page:

1. Click the ⚙️ icon next to "About"
2. Add topics:
   - `plant-disease-detection`
   - `deep-learning`
   - `computer-vision`
   - `fastapi`
   - `react`
   - `pytorch`
   - `machine-learning`
   - `ai`
   - `agriculture`
   - `plant-health`

## Step 7: Enable GitHub Pages (Optional)

For documentation hosting:

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** → **/ (root)** or **/docs**
4. Click **Save**

## Step 8: Add Repository Description

On the main repository page:

1. Click ⚙️ next to "About"
2. Add description: `AI-Powered Plant Disease Detection System using Deep Learning and NLP`
3. Add website (if deployed)
4. Add topics (see Step 6)
5. Click **Save changes**

## Step 9: Create Release (Optional)

1. Go to **Releases** → **Create a new release**
2. Tag version: `v1.0.0`
3. Release title: `PlantDocBot v1.0.0 - Initial Release`
4. Description: Copy from docs or write release notes
5. Click **Publish release**

## Common Issues & Solutions

### Issue: "Repository not found"
**Solution:** Check your username and repository name are correct

### Issue: "Permission denied"
**Solution:** 
- Use HTTPS instead of SSH, or
- Set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Issue: "Large files detected"
**Solution:** 
- Remove large files from git history
- Use Git LFS or upload separately
- See Step 4 above

### Issue: "Failed to push"
**Solution:**
- Pull first: `git pull origin main --rebase`
- Then push: `git push origin main`

## Useful Git Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Pull latest changes
git pull origin main

# Push changes
git push origin main

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View remote URL
git remote -v

# Change remote URL
git remote set-url origin https://github.com/NEW_USERNAME/plantdocbot.git
```

## Next Steps

After pushing to GitHub:

1. ✅ Add a professional profile picture for your GitHub account
2. ✅ Star your own repository (why not? 😄)
3. ✅ Share on LinkedIn/Twitter
4. ✅ Add to your resume/portfolio
5. ✅ Deploy to production (see docs/DEPLOYMENT.md)
6. ✅ Add CI/CD pipeline (optional)
7. ✅ Set up issue templates
8. ✅ Add code of conduct

## Keeping Your Repository Updated

```bash
# Make changes to your code
# ...

# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve bug"
# or
git commit -m "docs: update documentation"

# Push to GitHub
git push origin main
```

### Commit Message Conventions

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

---

## 🎉 Congratulations!

Your PlantDocBot is now on GitHub! 🌿

**Repository URL:** `https://github.com/YOUR_USERNAME/plantdocbot`

---

**Need Help?**
- [GitHub Docs](https://docs.github.com)
- [Git Documentation](https://git-scm.com/doc)
- Open an issue in your repository

**Happy Coding! 💻**
