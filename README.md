# Image Classification via RGB Histogram Similarity

A Python-based image classification system that categorizes images using RGB color histogram analysis and cosine similarity. Developed as a university project for the Parallel Algorithms course.

---

## How It Works

1. Each image's RGB channels are converted into **normalized histograms** (11 bins per channel)
2. Histograms are averaged per class to build a **class profile**
3. A query image is classified by computing **cosine similarity** against all class profiles
4. The class with the highest similarity score is the predicted label

---

## Tech Stack

- **Python 3** — core language
- **Pillow** — image loading and RGB conversion
- **NumPy** — array operations and cosine similarity computation
- **Matplotlib** — RGB histogram visualization
- **functools.reduce** — histogram aggregation and accumulation
- **itertools.groupby** — grouping images by class

> The entire pipeline is implemented using a **functional programming** style — `map`, `reduce`, `filter`, and `groupby` replace all explicit loops.

---

## Classes

Images are organized into three categories:

| Class | Examples |
|-------|---------|
| `animal` | cow, raccoon, chihuahua |
| `beauty` | Van Gogh, Mona Lisa, portrait |
| `pejzaz` | forest, mountains, river |

---

## Output

- Per-image **RGB histogram plots** (normalized frequency per bin)
- Classification result with **similarity score** for each image
- Summary of the **most similar image pair** across the dataset

---

## Getting Started
```bash
pip install pillow numpy matplotlib
python main.py
```

> Update the image paths in `list_class` at the top of `main.py` to point to your local files.
---

## Authors
Anastasija Jovanovic
Milica Jocic
