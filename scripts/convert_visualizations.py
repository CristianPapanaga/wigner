"""Convert the PDF visualizations in src/visualizations/ to PNG images in assets/."""

import os

import fitz

SRC = os.path.join("src", "visualizations")
DST = "assets"

FILES = [
    "Energy (per site) annealing plot.pdf",
    "Lattice with spin vectors.pdf",
    "Lattice.pdf",
    "Specific heat plot.pdf",
]

os.makedirs(DST, exist_ok=True)

for f in FILES:
    doc = fitz.open(os.path.join(SRC, f))
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    out_name = f.replace(".pdf", ".png")
    pix.save(os.path.join(DST, out_name))
    print(f"Converted {f} -> {out_name} ({pix.width}x{pix.height})")
    doc.close()