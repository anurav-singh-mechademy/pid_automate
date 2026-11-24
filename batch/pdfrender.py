"""PDF page rendering with PDFium (pypdfium2; BSD-3 / Apache-2.0). Model-free, shared by the pipeline, the batch
inventory, the precision filter and the model image's prepare endpoint. Pages come out as displayed: the PDF's own
/Rotate is applied. Bitmap size follows MuPDF's rounding (PDFium rounds up, so a 7200.0001 px side would give 7201);
the pipeline's pixel geometry was built on that size.
PDFium is not thread-safe: every call into it holds PDFIUM_LOCK (pdftext.py takes the same lock).
`path` may be a file path or the PDF's bytes."""
import math
import threading

import pypdfium2 as pdfium
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # 300 dpi A0 sheets are past PIL's decompression-bomb guard
PDFIUM_LOCK = threading.RLock()


def _source(path):
    return path if isinstance(path, (bytes, bytearray)) else str(path)


def page_count(path):
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(_source(path))
        try:
            return len(doc)
        finally:
            doc.close()


def render_page(path, page_no=0, dpi=300, gray=False):
    """PIL image of one page ('RGB', or 'L' when gray), annotations drawn."""
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(_source(path))
        try:
            pg = doc[page_no]
            im = pg.render(scale=dpi / 72, grayscale=gray, draw_annots=True).to_pil()
            w, h = pg.get_size()                                   # displayed size in points
        finally:
            doc.close()
    tw, th = math.ceil(w * dpi / 72 - 0.001), math.ceil(h * dpi / 72 - 0.001)
    if im.size != (tw, th):
        im = im.crop((0, 0, tw, th))
    return im.convert("L" if gray else "RGB")
