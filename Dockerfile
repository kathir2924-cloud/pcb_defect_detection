FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best.pt .

EXPOSE 7860

CMD ["python", "app.py"]
```

---

### Step 4 — Save It

1. Click the green **"Commit changes"** button
2. A popup appears — click **"Commit changes"** again

---

### Step 5 — Go to Render and Redeploy

1. Go to **render.com**
2. Click on your **pcb-defect-detection** service
3. Click **"Manual Deploy"** → **"Deploy latest commit"**
4. Watch the build log

---

### How to Confirm the New File is Saved

After committing, click on **Dockerfile** in your repo again and check the first few lines. It must show:
```
FROM python:3.10-slim
```
and
```
libgl1 \
