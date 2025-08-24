# # Sets up the Mistral client for making API requests.
import os
from pathlib import Path
import json, os, time
# Imports robustes selon version du SDK
try:
    from mistralai import FileChunk
except Exception:
    from mistralai.models import FileChunk


# --- Imports robustes pour les chunks ---
try:
    from mistralai import TextChunk
except Exception:
    from mistralai.models import TextChunk    

import base64, io

# Import robuste pour ImageURLChunk (au cas où)
try:
    from mistralai import ImageURLChunk
except Exception:
    from mistralai.models import ImageURLChunk

# Dépendances imagerie (fallback propre si absentes)
try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = None
    ImageOps = None
    ImageFilter = None

try:
    import cv2, numpy as np
except Exception:
    cv2 = None



api_key = os.environ.get('MdYKya1ABtT13jDBfYLWQGnK8GC5C7dl')
# Initialize Mistral client with API key
from mistralai import Mistral
client = Mistral(api_key='MdYKya1ABtT13jDBfYLWQGnK8GC5C7dl')

                          #### la partie pour les fichier pdf 

# ===================== OPTIONS =====================
PDF_PATH = r"C:/Users/aliou/OneDrive/Desktop/Mes projets/document_OCR_LLM/compromis-ANSELME_opt.pdf"
INCLUDE_IMAGE_BASE64 = True   # Mets False si tu n'as pas besoin des images -> réponses plus légères
FLATTEN_ANNOTATIONS = True    # Aplatir stylo/tampons/annotations avant OCR (recommandé)
MAX_RETRIES = 3               # Réessais côté client (réseau/API)
BACKOFF_BASE = 0.75           # secondes (exponentiel: 0.75s, 1.5s, 3s)
# ===================================================

# (Optionnel) exposer le chemin source pour tes fonctions d'affichage/markdown
ORIGINAL_PDF_PATH = PDF_PATH  # utilisé par ton get_combined_markdown()

# ---------- Utils ----------
def call_with_backoff(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            delay = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"[WARN] {fn.__name__} tentative {attempt} échouée: {e}. Retry dans {delay:.2f}s...")
            time.sleep(delay)

def _extract_file_id(up):
    for attr in ("id", "file_id"):
        v = getattr(up, attr, None)
        if v:
            return v
    inner = getattr(up, "file", None)
    if inner is not None:
        v = getattr(inner, "id", None)
        if v:
            return v
    if isinstance(up, dict):
        return up.get("id") or up.get("file_id") or (up.get("file") or {}).get("id")
    return None

def flatten_pdf_if_needed(pdf_in: Path) -> Path:
    """Aplatit les annotations (stylo/tampons) en rasterisant le PDF. Ignoré si PyMuPDF indisponible."""
    if not FLATTEN_ANNOTATIONS:
        return pdf_in
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        print(f"[WARN] Flatten ignoré (PyMuPDF manquant: {e})")
        return pdf_in

    try:
        doc = fitz.open(str(pdf_in))
        out = pdf_in.with_name(pdf_in.stem + "_flattened.pdf")
        zoom = 300 / 72.0
        mat = fitz.Matrix(zoom, zoom)
        out_doc = fitz.open()
        for page in doc:
            pix = page.get_pixmap(matrix=mat, annots=True, alpha=False)
            img = fitz.open("png", pix.tobytes("png"))
            rect = img[0].rect
            pdfbytes = img.convert_to_pdf()
            imgpdf = fitz.open("pdf", pdfbytes)
            newp = out_doc.new_page(width=rect.width, height=rect.height)
            newp.show_pdf_page(rect, imgpdf, 0)
        out_doc.save(str(out))
        out_doc.close()
        doc.close()
        print(f"[INFO] PDF aplati: {out}")
        return out
    except Exception as e:
        print(f"[WARN] Flatten échoué, on continue sur l'original: {e}")
        return pdf_in

# ---------- Main ----------
pdf_file = Path(PDF_PATH)
assert pdf_file.is_file(), f"Fichier introuvable: {pdf_file}"

pdf_for_ocr = flatten_pdf_if_needed(pdf_file)

# Upload (avec backoff)
uploaded_file = call_with_backoff(
    client.files.upload,
    file={"file_name": pdf_for_ocr.name, "content": pdf_for_ocr.read_bytes()},
    purpose="ocr",
    retries=3,  # si supporté par ta version du SDK
)
print("[INFO] Upload OK:", uploaded_file)

# file_id
file_id = _extract_file_id(uploaded_file)
assert file_id, f"file_id vide depuis la réponse: {uploaded_file}"
print("[INFO] file_id:", file_id)

# Vérification côté serveur (utile pour diagnostiquer les 404)
info = call_with_backoff(client.files.retrieve, file_id=file_id)
print(f"[INFO] Fichier côté serveur: {getattr(info, 'filename', None)} ; purpose={getattr(info, 'purpose', None)}")

# OCR (FileChunk direct) avec backoff
pdf_response = call_with_backoff(
    client.ocr.process,
    document=FileChunk(file_id=file_id),
    model="mistral-ocr-latest",
    include_image_base64=INCLUDE_IMAGE_BASE64,
)

# Sortie JSON (dict direct)
resp_dict = pdf_response.model_dump()
#print(json.dumps(resp_dict, indent=2)[:1000])  # aperçu

# Sauvegardes utiles
#out_json = pdf_for_ocr.with_suffix(".ocr.json")
#with open(out_json, "w", encoding="utf-8") as f:
#    json.dump(resp_dict, f, ensure_ascii=False, indent=2)
#print(f"[INFO] JSON complet écrit: {out_json}")


                      #### Affichage du resultat json pour les pdf

# --- Récupérer le markdown OCR (toutes les pages si dispo) ---
# Utilise pdf_response (objet SDK) OU resp_dict (dict) selon ce que tu as en mémoire.
def _build_pdf_ocr_markdown(pdf_response=None, resp_dict=None) -> str:
    # Cas 1 : objet SDK
    pages_obj = getattr(pdf_response, "pages", None)
    if pages_obj:
        return "\n\n---\n\n".join((getattr(p, "markdown", "") or "") for p in pages_obj).strip()
    # Cas 2 : dict (model_dump)
    if isinstance(resp_dict, dict):
        pages = resp_dict.get("pages") or []
        if pages:
            return "\n\n---\n\n".join((p.get("markdown") or "") for p in pages).strip()
    return ""

pdf_ocr_markdown = _build_pdf_ocr_markdown(pdf_response=pdf_response, resp_dict=resp_dict)
assert pdf_ocr_markdown, "OCR n'a renvoyé aucun markdown (pages vides)."

# --- Appel LLM pour convertir le markdown OCR en JSON structuré ---
chat_response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=[
        {
            "role": "user",
            "content": [
                TextChunk(
                    text=(
                        "Voici le résultat OCR d’un document (image) au format Markdown :\n\n"
                        f"{pdf_ocr_markdown}\n\n"
                        "Retourne STRICTEMENT un OBJET JSON valide (pas de tableau à la racine), sans aucun texte hors JSON. "
                        "Ne présuppose aucun schéma : utilise les titres/sections/étiquettes visibles dans le document comme clés, "
                        "et organise les informations de manière HIÉRARCHIQUE (évite les listes plates de paires 'field'/'value'). "
                        "Inclue autant d’informations que possible "
                        "en reprenant les libellés tels qu’imprimés. "
                        "Quand c’est pertinent, indique la page d’origine pour chaque élément, et représente les tableaux naturellement "
                        "(colonnes + lignes). "
                        "N’invente rien : ne renvoie que ce qui figure réellement dans le texte OCR. "
                        "En cas de doublons, déduplique et garde la version la plus complète. "
                        "La sortie doit être un OBJET JSON unique, propre, lisible et auto-descriptif."
                    )
                ),
            ],
        }
    ],
    response_format={"type": "json_object"},
    temperature=0,
)

# --- Récupération robuste du JSON renvoyé ---
# Selon le SDK, le contenu peut être une chaîne ou une liste de segments (chunks)
content = None
try:
    content = chat_response.choices[0].message.content
except Exception:
    # Repli au cas où certaines versions exposent d'autres attributs
    content = getattr(chat_response, "output_text", None) or getattr(chat_response, "content", None)

if isinstance(content, list):
    # Concatène les parties textuelles si le SDK renvoie une liste de segments
    texts = []
    for c in content:
        t = getattr(c, "text", None)
        if t:
            texts.append(t)
        elif isinstance(c, dict) and c.get("text"):
            texts.append(c["text"])
    content = "".join(texts)

content = (content or "").strip()

# Si le modèle a ajouté du texte autour, on isole la portion JSON entre le 1er { ou [ et le dernier } ou ]
if not (content.startswith("{") or content.startswith("[")):
    starts = [pos for pos in (content.find("{"), content.find("[")) if pos != -1]
    ends = [pos for pos in (content.rfind("}"), content.rfind("]")) if pos != -1]
    assert starts and ends, f"Réponse non-JSON : {content!r}"
    content = content[min(starts):max(ends)+1]

# Parse JSON ; si c'est un tableau, on l'emballe dans un objet standard
parsed = json.loads(content)
response_dict = {"items": parsed} if isinstance(parsed, list) else parsed

print(json.dumps(response_dict, indent=4, ensure_ascii=False)) 
 
                                           ####  la partie image  




# Verify image exists locally
#image_file = Path(r"C:/Users/aliou/OneDrive/Desktop/Mes projets/document_OCR_LLM/AVIS-IMPOT-ANSELME-REV-2023.jpg")
image_file=  Path(r"C:/Users/aliou/OneDrive/Desktop/Mes projets/document_OCR_LLM/image/concat.png")

assert image_file.is_file()

# ---- Prétraitement qualité (rotation EXIF, deskew, contraste, débruitage, sharpening, PNG) ----
def _preprocess_to_bytes(pth: Path):
    # Si PIL indisponible, retourner l'original
    if Image is None:
        return pth.read_bytes(), "image/jpeg"

    im = Image.open(pth).convert("RGB")
    # Rotation EXIF
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass

    if cv2 is not None:
        # OpenCV: deskew + éclairage + contraste + débruitage + sharpening
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Deskew (Hough)
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
            if lines is not None:
                angs = []
                for rho, theta in lines[:,0]:
                    ang = (theta * 180 / np.pi) - 90
                    if -45 < ang < 45:
                        angs.append(ang)
                if angs:
                    ang = float(np.median(angs))
                    if abs(ang) > 0.3 and abs(ang) <= 7.0:
                        (h, w) = bgr.shape[:2]
                        M = cv2.getRotationMatrix2D((w//2, h//2), -ang, 1.0)
                        cos, sin = abs(M[0,0]), abs(M[0,1])
                        nW = int((h*sin) + (w*cos)); nH = int((h*cos) + (w*sin))
                        M[0,2] += (nW/2) - (w//2); M[1,2] += (nH/2) - (h//2)
                        bgr = cv2.warpAffine(bgr, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass

        # Dé-glare (flash) + normalisation d'ombres
        try:
            _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
            if mask.mean() >= 1:
                bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            dil = cv2.dilate(gray, np.ones((15,15), np.uint8))
            bg = cv2.medianBlur(dil, 35)
            gray = cv2.divide(gray, bg, scale=255)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

        # CLAHE + débruitage + unsharp
        try:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            y = clahe.apply(y)
            bgr = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_blur = cv2.GaussianBlur(y, (3,3), 0)
            y_sharp = cv2.addWeighted(y, 1.2, y_blur, -0.2, 0)
            bgr = cv2.cvtColor(cv2.merge([y_sharp, cr, cb]), cv2.COLOR_YCrCb2BGR)
        except Exception:
            pass

        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        # Fallback PIL simple
        try:
            im = ImageOps.autocontrast(im)
        except Exception:
            pass
        try:
            im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        except Exception:
            pass

    # Upscale si trop petit
    try:
        if im.height < 1500:
            scale = 1500 / im.height
            im = im.resize((int(im.width * scale), 1500), resample=Image.LANCZOS)
    except Exception:
        pass

    # PNG lossless pour meilleurs détails fins
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"

# Encode the image in base64 format, required for API consumption
img_bytes, mime = _preprocess_to_bytes(image_file)
encoded = base64.b64encode(img_bytes).decode()
base64_data_url = f"data:{mime};base64,{encoded}"

# Process the receipt image with OCR using the Mistral OCR model
image_response = client.ocr.process(
    document=ImageURLChunk(image_url=base64_data_url),
    model="mistral-ocr-latest",
    include_image_base64=True  # récupère aussi les images intégrées utiles (logos/QR)
)

# Convert the OCR response to a JSON format and print it
response_dict = json.loads(image_response.model_dump_json())
json_string = json.dumps(response_dict, indent=4)
#print(json_string)


                         #### Affichage du resultat en json ppour les images



import json

# --- Imports robustes pour les chunks ---
try:
    from mistralai import ImageURLChunk, TextChunk
except Exception:
    from mistralai.models import ImageURLChunk, TextChunk

# --- Récupérer le markdown OCR (toutes les pages si dispo) ---
pages = getattr(image_response, "pages", None) or []
assert pages, "OCR n'a renvoyé aucune page."
image_ocr_markdown = "\n\n---\n\n".join([(getattr(p, "markdown", "") or "") for p in pages]).strip()

# Get structured response from model
chat_response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=[
        {
            "role": "user",
            "content": [
                TextChunk(
                    text=(
                        "Voici le résultat OCR d’un document (image) au format Markdown :\n\n"
                        f"{image_ocr_markdown}\n\n"
                        "Retourne STRICTEMENT un OBJET JSON valide (pas de tableau à la racine), sans aucun texte hors JSON. "
                        "Ne présuppose aucun schéma : utilise les titres/sections/étiquettes visibles dans le document comme clés, "
                        "et organise les informations de manière HIÉRARCHIQUE (évite les listes plates de paires 'field'/'value'). "
                        "Inclue autant d’informations que possible"
                        "Quand c’est pertinent, indique la page d’origine pour chaque élément, et représente les tableaux naturellement "
                        "(colonnes + lignes). "
                        "N’invente rien : ne renvoie que ce qui figure réellement dans le texte OCR. "
                        "En cas de doublons, déduplique et garde la version la plus complète. "
                        "La sortie doit être un OBJET JSON unique, propre, lisible et auto-descriptif."
                    )
                ),
            ],
        }
    ],
    response_format={"type": "json_object"},
    temperature=0,
)


# --- Récupération robuste du JSON renvoyé ---
# Selon le SDK, le contenu peut être une chaîne ou une liste de segments (chunks)
content = None
try:
    content = chat_response.choices[0].message.content
except Exception:
    # Repli au cas où certaines versions exposent d'autres attributs
    content = getattr(chat_response, "output_text", None) or getattr(chat_response, "content", None)

if isinstance(content, list):
    # Concatène les parties textuelles si le SDK renvoie une liste de segments
    texts = []
    for c in content:
        t = getattr(c, "text", None)
        if t:
            texts.append(t)
        elif isinstance(c, dict) and c.get("text"):
            texts.append(c["text"])
    content = "".join(texts)

content = (content or "").strip()

# Si le modèle a ajouté du texte autour, on isole la portion JSON entre le 1er { ou [ et le dernier } ou ]
if not (content.startswith("{") or content.startswith("[")):
    starts = [pos for pos in (content.find("{"), content.find("[")) if pos != -1]
    ends = [pos for pos in (content.rfind("}"), content.rfind("]")) if pos != -1]
    assert starts and ends, f"Réponse non-JSON : {content!r}"
    content = content[min(starts):max(ends)+1]

# Parse JSON ; si c'est un tableau, on l'emballe dans un objet standard
parsed = json.loads(content)
response_dict = {"items": parsed} if isinstance(parsed, list) else parsed

print(json.dumps(response_dict, indent=4, ensure_ascii=False))










