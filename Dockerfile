FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
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

**8.** After pasting, look at line 6 — it must show:
```
    libgl1 \
```
NOT `libgl1-mesa-glx` — if you see `mesa` the old text is still there

**9.** Click the green **"Commit changes"** button (top right)

**10.** A small popup appears — click **"Commit changes"** again

---

### Then Go to Render

**11.** Go to **render.com** → your service

**12.** Click **"Manual Deploy"** → **"Deploy latest commit"**

**13.** Watch the logs — the error line should be completely gone now

---

### Quick Check Before Deploying

After saving on GitHub, click **Dockerfile** again and confirm line 7 shows exactly:
```
    libgl1 \
