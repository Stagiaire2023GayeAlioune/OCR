# app.py
import streamlit as st
import tempfile
from pathlib import Path
import os, json, time, base64, io

st.set_page_config(page_title="OCR PDF/Image → JSON (Mistral)", layout="wide")
st.title("📄🖼️ OCR PDF / Image → JSON (Mistral)")
st.caption("Upload un PDF ou une image : l’app route automatiquement vers la partie correspondante et affiche le JSON produit.")

# ===================== UPLOAD =====================
uploaded = st.file_uploader(
    "Importer un fichier (PDF ou image PNG/JPG/JPEG)",
    type=["pdf", "png", "jpg", "jpeg"]
)

if not uploaded:
    st.info("👉 Importez un PDF ou une image pour commencer.")
    st.stop()

# On écrit le fichier uploadé dans un fichier temporaire
suffix = "." + uploaded.name.split(".")[-1].lower()
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(uploaded.read())
tmp.flush()
tmp_path = Path(tmp.name)
st.success(f"Fichier reçu : **{uploaded.name}**")
st.caption(f"Chemin temporaire : `{tmp_path}`")

# ===================== COMMUN : Imports tels quels =====================
# Sets up the Mistral client for making API requests.
try:
    from mistralai import FileChunk
except Exception:
    try:
        from mistralai.models import FileChunk
    except Exception as e:
        st.error(f"❌ Erreur d'import FileChunk: {e}")
        st.stop()

# --- Imports robustes pour les chunks ---
try:
    from mistralai import TextChunk
except Exception:
    from mistralai.models import TextChunk    

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
    np = None

# ⚠️ Clé Mistral : configuration
from mistralai import Mistral
api_key = os.environ.get('MISTRAL_API_KEY', 'MdYKya1ABtT13jDBfYLWQGnK8GC5C7dl')

if not api_key or api_key == 'your-api-key-here':
    st.error("❌ Clé API Mistral manquante. Définissez la variable d'environnement MISTRAL_API_KEY.")
    st.info("💡 Vous pouvez aussi modifier directement la clé dans le code (ligne 62)")
    st.stop()

try:
    client = Mistral(api_key=api_key)
    st.success("✅ Connexion Mistral établie")
except Exception as e:
    st.error(f"❌ Erreur de connexion Mistral: {e}")
    st.stop()

# ===================== ROUTAGE SELON EXTENSION =====================
is_pdf = suffix == ".pdf"
is_img = suffix in {".png", ".jpg", ".jpeg"}

# -------------------------------------------------
# ----------------- PARTIE PDF --------------------
# -------------------------------------------------
if is_pdf:
    st.subheader("📄 Traitement PDF")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # ===================== OPTIONS =====================
    PDF_PATH = str(tmp_path)  # <--- on ne change QUE le chemin
    INCLUDE_IMAGE_BASE64 = True
    FLATTEN_ANNOTATIONS = True
    MAX_RETRIES = 3
    BACKOFF_BASE = 0.75
    # ===================================================

    ORIGINAL_PDF_PATH = PDF_PATH

    # ---------- Utils ----------
    def call_with_backoff(fn, *args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                delay = BACKOFF_BASE * (2 ** (attempt - 1))
                st.warning(f"[WARN] {fn.__name__} tentative {attempt} échouée: {e}. Retry dans {delay:.2f}s...")
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
            st.warning(f"[WARN] Flatten ignoré (PyMuPDF manquant: {e})")
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
            st.info(f"[INFO] PDF aplati: {out}")
            return out
        except Exception as e:
            st.warning(f"[WARN] Flatten échoué, on continue sur l'original: {e}")
            return pdf_in

    # ---------- Main ----------
    pdf_file = Path(PDF_PATH)
    assert pdf_file.is_file(), f"Fichier introuvable: {pdf_file}"

    pdf_for_ocr = flatten_pdf_if_needed(pdf_file)

    # Upload (avec backoff)
    status_text.text("📤 Upload du fichier PDF...")
    progress_bar.progress(20)
    
    uploaded_file = call_with_backoff(
        client.files.upload,
        file={"file_name": pdf_for_ocr.name, "content": pdf_for_ocr.read_bytes()},
        purpose="ocr",
        retries=3,
    )
    st.write("[INFO] Upload OK:", uploaded_file)

    # file_id
    file_id = _extract_file_id(uploaded_file)
    assert file_id, f"file_id vide depuis la réponse: {uploaded_file}"
    st.write("[INFO] file_id:", file_id)

    # Vérification côté serveur
    info = call_with_backoff(client.files.retrieve, file_id=file_id)
    st.write(f"[INFO] Fichier côté serveur: {getattr(info, 'filename', None)} ; purpose={getattr(info, 'purpose', None)}")

    # OCR (FileChunk direct) avec backoff
    status_text.text("🔍 Traitement OCR en cours...")
    progress_bar.progress(60)
    
    pdf_response = call_with_backoff(
        client.ocr.process,
        document=FileChunk(file_id=file_id),
        model="mistral-ocr-latest",
        include_image_base64=INCLUDE_IMAGE_BASE64,
    )

    # Sortie JSON (dict direct)
    resp_dict = pdf_response.model_dump()

    # ---------------- Affichage du résultat JSON pour les PDF ----------------
    def _build_pdf_ocr_markdown(pdf_response=None, resp_dict=None) -> str:
        pages_obj = getattr(pdf_response, "pages", None)
        if pages_obj:
            return "\n\n---\n\n".join((getattr(p, "markdown", "") or "") for p in pages_obj).strip()
        if isinstance(resp_dict, dict):
            pages = resp_dict.get("pages") or []
            if pages:
                return "\n\n---\n\n".join((p.get("markdown") or "") for p in pages).strip()
        return ""

    pdf_ocr_markdown = _build_pdf_ocr_markdown(pdf_response=pdf_response, resp_dict=resp_dict)
    assert pdf_ocr_markdown, "OCR n'a renvoyé aucun markdown (pages vides)."

    status_text.text("🧠 Structuration JSON en cours...")
    progress_bar.progress(80)
    
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

    content = None
    try:
        content = chat_response.choices[0].message.content
    except Exception:
        content = getattr(chat_response, "output_text", None) or getattr(chat_response, "content", None)

    if isinstance(content, list):
        texts = []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                texts.append(t)
            elif isinstance(c, dict) and c.get("text"):
                texts.append(c["text"])
        content = "".join(texts)

    content = (content or "").strip()
    if not (content.startswith("{") or content.startswith("[")):
        starts = [pos for pos in (content.find("{"), content.find("[")) if pos != -1]
        ends = [pos for pos in (content.rfind("}"), content.rfind("]")) if pos != -1]
        assert starts and ends, f"Réponse non-JSON : {content!r}"
        content = content[min(starts):max(ends)+1]

    parsed = json.loads(content)
    response_dict = {"items": parsed} if isinstance(parsed, list) else parsed

    progress_bar.progress(100)
    status_text.text("✅ Traitement terminé !")
    
    st.subheader("🧠 JSON structuré (PDF)")
    st.json(response_dict)

# -------------------------------------------------
# ----------------- PARTIE IMAGE ------------------
# -------------------------------------------------
if is_img:
    st.subheader("🖼️ Traitement Image")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Verify image exists locally
    image_file = Path(str(tmp_path))  # <--- on ne change QUE le chemin

    assert image_file.is_file()

    # ---- Prétraitement qualité (rotation EXIF, deskew, contraste, débruitage, sharpening, PNG) ----
    def _preprocess_to_bytes(pth: Path):
        if Image is None:
            return pth.read_bytes(), "image/jpeg"

        im = Image.open(pth).convert("RGB")
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass

        if cv2 is not None:
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

            # Dé-glare + normalisation d'ombres
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

        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"

    status_text.text("🖼️ Prétraitement de l'image...")
    progress_bar.progress(30)
    
    img_bytes, mime = _preprocess_to_bytes(image_file)
    encoded = base64.b64encode(img_bytes).decode()
    base64_data_url = f"data:{mime};base64,{encoded}"

    # OCR image
    status_text.text("🔍 Traitement OCR de l'image...")
    progress_bar.progress(60)
    
    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest",
        include_image_base64=True
    )

    response_dict_raw = json.loads(image_response.model_dump_json())

    # ---------------- Affichage du résultat JSON pour les images ----------------
    # (imports robustes déjà faits plus haut)
    pages = getattr(image_response, "pages", None) or []
    assert pages, "OCR n'a renvoyé aucune page."
    image_ocr_markdown = "\n\n---\n\n".join([(getattr(p, "markdown", "") or "") for p in pages]).strip()

    status_text.text("🧠 Structuration JSON en cours...")
    progress_bar.progress(80)
    
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

    content = None
    try:
        content = chat_response.choices[0].message.content
    except Exception:
        content = getattr(chat_response, "output_text", None) or getattr(chat_response, "content", None)

    if isinstance(content, list):
        texts = []
        for c in content:
            t = getattr(c, "text", None)
            if t:
                texts.append(t)
            elif isinstance(c, dict) and c.get("text"):
                texts.append(c["text"])
        content = "".join(texts)

    content = (content or "").strip()
    if not (content.startswith("{") or content.startswith("[")):
        starts = [pos for pos in (content.find("{"), content.find("[")) if pos != -1]
        ends = [pos for pos in (content.rfind("}"), content.rfind("]")) if pos != -1]
        assert starts and ends, f"Réponse non-JSON : {content!r}"
        content = content[min(starts):max(ends)+1]

    parsed = json.loads(content)
    response_dict = {"items": parsed} if isinstance(parsed, list) else parsed

    progress_bar.progress(100)
    status_text.text("✅ Traitement terminé !")
    
    st.subheader("🧠 JSON structuré (Image)")
    st.json(response_dict)

# -------------------------------------------------
# -------------- Garde-fous & UI ------------------
# -------------------------------------------------
if not (is_pdf or is_img):
    st.error("Extension non supportée. Importez un fichier .pdf, .png, .jpg ou .jpeg.")
