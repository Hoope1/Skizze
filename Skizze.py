import cv2
import numpy as np
import os
import sys

# Optional: skimage für Skelettierung
try:
    from skimage.morphology import skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def load_image(path: str) -> np.ndarray:
    """
    Lädt ein Bild von der Festplatte und gibt es als BGR-Array zurück.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Bild unter '{path}' konnte nicht geladen werden.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Konvertiert ein BGR-Bild in ein Graustufenbild.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def auto_threshold_otsu(image_gray: np.ndarray) -> (np.ndarray, float):
    """
    Führt Otsu-Schwellenwertbestimmung durch, gibt binäres Bild und genutzten Schwellwert zurück.
    """
    # Otsu-Schwellwert (zweistufig)
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    retval, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, retval


def adaptive_threshold(image_gray: np.ndarray, block_size: int = 51, c: int = 10) -> np.ndarray:
    """
    Führt adaptiven Schwellwert durch (Mean-Ansatz) und gibt binär-image zurück.
    - block_size: Größe des lokalen Gebiets, muss ungerade sein
    - c: Konstante, die vom Mittelwert abgezogen wird
    """
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, c
    )
    return binary


def detect_edges_canny(image_gray: np.ndarray, low_thresh: float = None, high_thresh: float = None) -> np.ndarray:
    """
    Erkennt Kanten mit dem Canny-Algorithmus. Falls keine Schwellwerte angegeben,
    werden sie automatisch basierend auf Median des Bildes berechnet.
    """
    if low_thresh is None or high_thresh is None:
        # Automatische Schätzung der Schwellwerte anhand des Bild-Medians
        v = np.median(image_gray)
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
    else:
        low, high = low_thresh, high_thresh
    edges = cv2.Canny(image_gray, low, high, apertureSize=3)
    return edges


def detect_lines_hough(edges: np.ndarray,
                        rho: float = 1,
                        theta: float = np.pi/180,
                        threshold: int = 100,
                        min_line_len: int = 50,
                        max_line_gap: int = 10) -> np.ndarray:
    """
    Führt probabilistische Hough-Transformation durch, gibt ein Array von Linien zurück.
    Jeder Eintrag ist (x1, y1, x2, y2).
    """
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def filter_lines_by_length(lines: np.ndarray, min_length: float = 50.0) -> np.ndarray:
    """
    Filtert erkannte Linien nach Länge (euklidische Distanz zwischen Endpunkten).
    """
    if lines.size == 0:
        return lines
    filtered = []
    for x1, y1, x2, y2 in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            filtered.append([x1, y1, x2, y2])
    return np.array(filtered, dtype=np.int32)


def extract_contours(image_bin: np.ndarray) -> list:
    """
    Extrahiert Konturen aus einem binären Bild und gibt sie als Liste von Punkten zurück.
    """
    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Optional: Vereinfachung der Konturen (Approximation)
    approx_contours = [cv2.approxPolyDP(cnt, epsilon=1.0, closed=False) for cnt in contours]
    return approx_contours


def skeletonize_image(image_bin: np.ndarray) -> np.ndarray:
    """
    Führt Skelettierung auf einem binären Bild durch (nur Schwarz/Weiß, 0/255).
    Benötigt skimage; falls nicht verfügbar, wird der Originalbinary zurückgegeben.
    """
    if not _SKIMAGE_AVAILABLE:
        print("skimage.morphology.skeletonize nicht verfügbar. Rückgabe des Originalbildes.")
        return image_bin
    # Skelettierungsfunktion von skimage erwartet Binärbild mit Werten 0/1
    bin01 = (image_bin > 0).astype(np.uint8)
    skeleton = skeletonize(bin01)
    # Rückkonvertierung zu 0/255
    return (skeleton * 255).astype(np.uint8)


def draw_lines_on_blank(image_shape: tuple, lines: np.ndarray, color: tuple = (255, 255, 255), thickness: int = 1) -> np.ndarray:
    """
    Zeichnet Linien auf ein leeres BGR-Bild der gegebenen Form (Höhe, Breite).
    """
    blank = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(blank, (x1, y1), (x2, y2), color, thickness)
    return blank


def draw_contours_on_blank(image_shape: tuple, contours: list, color: tuple = (255, 255, 255), thickness: int = 1) -> np.ndarray:
    """
    Zeichnet Konturen (als Polylinien) auf ein leeres BGR-Bild.
    """
    blank = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        for i in range(len(pts) - 1):
            cv2.line(blank, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
    return blank


def combine_line_images(*images: np.ndarray) -> np.ndarray:
    """
    Kombiniert mehrere BGR-Bilder per logischem ODER (pixelweise),
    sodass alle Linien aus allen Quellen zusammengeführt werden.
    Erwartet Farb- oder Graustufenbilder (konvertiert intern zu binär mask).
    """
    if len(images) == 0:
        raise ValueError("Mindestens ein Bild muss übergeben werden.")
    # Alle Bilder auf Graustufen und binärisieren
    h, w = images[0].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for img in images:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binärisierung: >0 bleiben
        bin_mask = (gray > 0).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, bin_mask)
    # Rückgabe als BGR für weitere Verarbeitung
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


def process_image(path: str,
                  output_dir: str = None,
                  use_otsu: bool = True,
                  adaptive_block: int = 51,
                  adaptive_c: int = 10,
                  hough_params: dict = None,
                  filter_min_length: float = 50.0,
                  do_skeleton: bool = False) -> dict:
    """
    Hauptpipeline, die alle Schritte ausführt:
    1. Einlesen und Graustufen
    2. Binärisierung (Otsu oder Adaptiv)
    3. Kantenerkennung (Canny)
    4. Linienerkennung (Hough)
    5. Konturenanalyse
    6. Optionale Skelettierung
    7. Zusammenführung und Export

    Gibt ein Dictionary mit allen Zwischenergebnissen zurück.
    """
    # 1. Einlesen
    img_color = load_image(path)
    img_gray = to_grayscale(img_color)

    # 2. Binärisierung
    if use_otsu:
        bin_img, thresh_val = auto_threshold_otsu(img_gray)
        print(f"Verwendeter Otsu-Schwellwert: {thresh_val}")
    else:
        bin_img = adaptive_threshold(img_gray, block_size=adaptive_block, c=adaptive_c)
        thresh_val = None
        print(f"Adaptive Schwellwert-Binarisierung mit block_size={adaptive_block}, c={adaptive_c}")

    # 3. Kantenerkennung
    edges = detect_edges_canny(img_gray)
    print("Kantenerkennung (Canny) abgeschlossen.")

    # 4. Linienerkennung (Hough)
    if hough_params is None:
        hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 100,
            "min_line_len": 50,
            "max_line_gap": 10
        }
    lines = detect_lines_hough(edges,
                               rho=hough_params["rho"],
                               theta=hough_params["theta"],
                               threshold=hough_params["threshold"],
                               min_line_len=hough_params["min_line_len"],
                               max_line_gap=hough_params["max_line_gap"])
    print(f"Hough-Erkennung: {len(lines)} Linien gefunden.")

    # Filtere Linien nach Länge
    lines = filter_lines_by_length(lines, min_length=filter_min_length)
    print(f"Nach Längenfilter: {len(lines)} Linien übrig.")

    # 5. Konturenanalyse
    contours = extract_contours(bin_img)
    print(f"Kontur-Extraktion: {len(contours)} Konturen gefunden.")

    # 6. Optionale Skelettierung
    if do_skeleton:
        skeleton = skeletonize_image(bin_img)
        print("Skelettierung abgeschlossen.")
    else:
        skeleton = None

    # 7. Zeichnen und Zusammenführen
    h, w = img_gray.shape
    lines_img = draw_lines_on_blank((h, w), lines, color=(255, 255, 255), thickness=1)
    contours_img = draw_contours_on_blank((h, w), contours, color=(255, 255, 255), thickness=1)

    if skeleton is not None:
        skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        combined = combine_line_images(lines_img, contours_img, skeleton_bgr)
    else:
        combined = combine_line_images(lines_img, contours_img)

    # Ausgabe
    results = {
        "original_color": img_color,
        "grayscale": img_gray,
        "binary": bin_img,
        "edges": edges,
        "lines": lines,
        "contours": contours,
        "skeleton": skeleton,
        "lines_image": lines_img,
        "contours_image": contours_img,
        "combined": combined,
        "threshold_value": thresh_val
    }

    # Optional: Speichern
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Speichere Zwischenergebnisse
        cv2.imwrite(os.path.join(output_dir, "grayscale.png"), img_gray)
        cv2.imwrite(os.path.join(output_dir, "binary.png"), bin_img)
        cv2.imwrite(os.path.join(output_dir, "edges.png"), edges)
        cv2.imwrite(os.path.join(output_dir, "lines.png"), lines_img)
        cv2.imwrite(os.path.join(output_dir, "contours.png"), contours_img)
        cv2.imwrite(os.path.join(output_dir, "combined.png"), combined)
        if skeleton is not None:
            cv2.imwrite(os.path.join(output_dir, "skeleton.png"), skeleton)
        print(f"Bilder in '{output_dir}' gespeichert.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimale Linien zum Abpausen aus einem Bild extrahieren.")
    parser.add_argument("--input", required=True, help="Pfad zum Eingabebild.")
    parser.add_argument("--output-dir", default=None, help="Verzeichnis für Zwischenergebnisse.")
    parser.add_argument("--no-otsu", action="store_true", help="Deaktiviere Otsu, benutze stattdessen adaptive Binarisierung.")
    parser.add_argument("--adaptive-block", type=int, default=51, help="Blockgröße für adaptive Binarisierung (ungerade).")
    parser.add_argument("--adaptive-c", type=int, default=10, help="Konstante C für adaptive Binarisierung.")
    parser.add_argument("--min-line-len", type=float, default=50.0, help="Minimale Länge für Hough-Linien.")
    parser.add_argument("--do-skeleton", action="store_true", help="Führe zusätzliche Skelettierung durch.")
    args = parser.parse_args()

    results = process_image(
        path=args.input,
        output_dir=args.output_dir,
        use_otsu=not args.no_otsu,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        filter_min_length=args.min_line_len,
        do_skeleton=args.do_skeleton
    )
    print("Verarbeitung abgeschlossen. Die kombinierten Linien wurden erzeugt.")
