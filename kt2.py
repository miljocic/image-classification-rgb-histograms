import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
from itertools import groupby

# Konstantna vrednost koja definiše broj binova (segmenata) za histogram.
NUM_BINS = 11

list_class = [
    ('animal', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/animal/cow.jpg'),
    ('animal', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/animal/racon.jpg'),
    ('animal', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/animal/civauva.png'),
    ('beauty', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/beauty/vangog.jpg'),
    ('beauty', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/beauty/monalisa.jpg'),
    ('beauty', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/beauty/zena.jpg'),
    ('pejzaz', 'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/pejzaz/reka.jpg'),
    ('pejzaz',
     'C:/Users/munja/PycharmProjects/p24-25-drugi-projekat-tim_mjocic10723_ajovanovic11423/resours/pejzaz/suma.jpg')
]

# Funkcija za izračunavanje histograma RGB kanala jedne slike.
def calculate_histograms(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')

    except FileNotFoundError:
        print(f"Image {image_path} not found.")
        return None

    # Slike pretvaramo u Numpy niz
    img_array = np.array(image)

    # Funkcija za mapiranje vrednosti piksela (0-255) u odgovarajući bin (0 do NUM_BINS-1).
    def get_bin(value):
        return (int(value) * NUM_BINS) // 256

    # Izravnava 3D niz (visina × širina × 3) u 2D niz (broj_piksela × 3).
    pixels = img_array.reshape(-1, 3)

    r_bins = list(map(lambda pixel: get_bin(pixel[0]), pixels))
    g_bins = list(map(lambda pixel: get_bin(pixel[1]), pixels))
    b_bins = list(map(lambda pixel: get_bin(pixel[2]), pixels))

    # Funkcija za ažuriranje histograma za određeni bin.
    def histogram_reducer(hist, bin_value):
        hist[bin_value] = hist.get(bin_value, 0) + 1
        return hist

    hist_r = reduce(histogram_reducer, r_bins, {})
    hist_g = reduce(histogram_reducer, g_bins, {})
    hist_b = reduce(histogram_reducer, b_bins, {})

    total_pixels = img_array.shape[0] * img_array.shape[1]

    normalize = lambda hist: np.array(list(map(lambda i: hist.get(i, 0) / total_pixels, range(NUM_BINS))))
    hist_r = normalize(hist_r)
    hist_g = normalize(hist_g)
    hist_b = normalize(hist_b)

    return hist_r, hist_g, hist_b

# Funkcija za izračunavanje prosečnih histograma za svaku klasu slika.
def calculate_average_histograms(image_class_pairs):

    grouped_images = groupby(sorted(image_class_pairs, key=lambda x: x[0]), key=lambda x: x[0])

    # Agregacija histograma svih slika u grupi iste klase.
    def aggregate_histograms(images):
        histograms = list(map(lambda img: calculate_histograms(img[1]), images))
        count = reduce(lambda acc, _: acc + 1, histograms, 0)

        sum_hist = reduce(
            lambda acc, hists: (
                acc[0] + hists[0],
                acc[1] + hists[1],
                acc[2] + hists[2]
            ),
            histograms,
            (np.zeros(NUM_BINS), np.zeros(NUM_BINS), np.zeros(NUM_BINS))
        )

        # Vraća prosečne histograme za klasu (sabirke se dele s brojem slika).
        return tuple(map(lambda hist: hist / count, sum_hist))

    # Kreira listu prosečnih histograma za svaku klasu.
    return list(map(lambda pair: (pair[0], aggregate_histograms(pair[1])), grouped_images))

# Funkcija za klasifikaciju slike na osnovu prosečnih histograma klasa.
def classify_image(image_path, average_histograms):

    image_histogram = calculate_histograms(image_path)

    if image_histogram is None:
        return None

    similarities = []
    similarities = list(map(lambda avg: (avg[0], cosine_similarity(image_histogram, avg[1])), average_histograms))

    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    best_match = sorted_similarities[0]

    return os.path.basename(image_path), best_match[0], best_match[1]

# Funkcija za crtanje histograma za RGB kanale slike.
def plot_histogram(hist_r, hist_g, hist_b, image_name):

    bin_edges = list(
        map(lambda i: f"Bin {i + 1}: {i * (256 // NUM_BINS)}-{(i + 1) * (256 // NUM_BINS)}", range(NUM_BINS)))

    plt.figure(figsize=(10, 6))
    plt.plot(range(NUM_BINS), hist_r, color='red', label='Red Channel', linewidth=2)
    plt.plot(range(NUM_BINS), hist_g, color='green', label='Green Channel', linewidth=2)
    plt.plot(range(NUM_BINS), hist_b, color='blue', label='Blue Channel', linewidth=2)

    plt.title(f'RGB Channel Histograms za {image_name}')
    plt.xlabel('Bin Range')
    plt.ylabel('Normalized Frequency')
    plt.xticks(ticks=range(NUM_BINS), labels=bin_edges, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Funkcija za izračunavanje kosinusne sličnosti između dva histograma.
def cosine_similarity(hist1, hist2):

    hist1_flat = np.concatenate(hist1)
    hist2_flat = np.concatenate(hist2)

    dot_product = np.dot(hist1_flat, hist2_flat)
    norm1 = np.sqrt(np.dot(hist1_flat, hist1_flat))
    norm2 = np.sqrt(np.dot(hist2_flat, hist2_flat))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

# Funkcija za prikaz sličnosti između svih slika.
def summarize_similarities(image_histograms):
    print("\nSummary of Most Similar Images:")

    def compute_similarities(item):
        img1_id, (_, hist1) = item
        similarities = list(map(
            lambda img2_item: (img2_item[0], cosine_similarity(hist1, img2_item[1][1])),
            image_histograms.items()
        ))
        similarities = filter(lambda x: img1_id != x[0], similarities)
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        if sorted_similarities:
            most_similar_img, max_similarity = sorted_similarities[0]
            print(f"{img1_id} is most similar to {most_similar_img} with a similarity of {max_similarity:.4f}")

    list(map(compute_similarities, image_histograms.items()))


# Glavna funkcija koja obrađuje sve slike i prikazuje njihove rezultate.
def process_all_images(image_class_pairs):

    average_histograms = calculate_average_histograms(image_class_pairs)

    def process_image(pair):
        class_label, image_path = pair
        try:
            hist = calculate_histograms(image_path)

            if hist is not None:
                image_id = os.path.basename(image_path)
                image_histograms = {image_id: (class_label, hist)}

                hist_r, hist_g, hist_b = hist
                plot_histogram(hist_r, hist_g, hist_b, image_id)

                classification = classify_image(image_path, average_histograms)
                print(f"Image {classification[0]} classified as {classification[1]} with similarity {classification[2]:.4f}")

                return image_histograms

        except FileNotFoundError:
            print(f"File {image_path} not found.")

        except AttributeError as e:
            print(f"Failed to open {image_path}: {e}")

        # Ako dođe do izuzetka, vraća se prazan rečnik kako bi se izbeglo prekidanje toka programa.
        return {}


    image_histograms = reduce(
        lambda acc, pair: {**acc, **process_image(pair)},
        image_class_pairs,
        {}
    )

    summarize_similarities(image_histograms)

process_all_images(list_class)
