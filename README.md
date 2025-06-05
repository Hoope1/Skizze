# Skizze

Ein Werkzeug zur Extraktion klarer Linien aus Bildern. Die Verarbeitung umfasst Rauschunterdrückung, adaptive Binarisierung und optionale Deep-Learning-Methoden. Standardwerte können in `pyproject.toml` konfiguriert werden. Beim ersten Aufruf legt das Skript automatisch eine virtuelle Umgebung unter `env/` an und installiert fehlende Pakete. Dieses Verhalten kann mit `SKIZZE_SKIP_VENV=1` und `SKIZZE_SKIP_INSTALL=1` abgeschaltet werden.

```bash
python -m skizze.cli --input bild.png --output-dir out

Weitere Optionen siehe `--help`. Unter anderem lassen sich
lokales Thresholding, Multi-Scale-Binarisierung sowie Deep-Learning-
basierte Linienerkennung (DeepLSD oder FClip) aktivieren.
```
