Step 1 — Create the repo on GitHub
Go to github.com → click the + icon → New repository

Name it: chickpea-quality-assessment
Set to Public
Do NOT check "Add README" (you'll add it manually)
Click Create repository


Step 2 — Organise your project folder locally
Make sure your project folder looks something like this:
chickpea-quality-assessment/
├── chickpea_app.py
├── requirements.txt       ← create this
├── README.md              ← create this
└── (any other .py files)

Create requirements.txt — open terminal in your project folder and run:
bashpip freeze > requirements.txt
```

**Step 3 — Create README.md**

Open any text editor, create `README.md` with this content:
```
# Chickpea Quality Assessment Tool

AI-powered chickpea grain quality assessment using YOLOv8 and Vision Transformer.

## What it does
Automatically classifies chickpea seeds into quality categories (healthy, broken, 
discolored, large, small) from uploaded images using deep learning.

## Tech Stack
Python · PyTorch · Hugging Face Transformers · YOLOv8 · Streamlit · OpenCV · Plotly

## How to run
pip install -r requirements.txt
streamlit run chickpea_app.py

## Model
- YOLOv8m trained on TRCS_8_SET dataset (7,200 images, 8 classes)
- Vision Transformer (ViT) via Hugging Face for quality classification
- mAP@0.5: 0.97 at epoch 300

Step 4 — Push to GitHub
Open terminal inside your project folder and run these commands one by one:
bashgit init
git add .
git commit -m "Initial commit — chickpea quality assessment tool"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/chickpea-quality-assessment.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

**Step 5 — Pin it on your profile**

Go to your GitHub profile page → click **Customize your pins** → select `chickpea-quality-assessment` → save.

---

**One thing to check before pushing:**

If your code has any API keys, model weights, or large files (anything over 50MB) — don't push those. Create a `.gitignore` file first:
```
*.pt
*.weights
__pycache__/
.env
*.h5
---