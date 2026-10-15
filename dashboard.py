from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from acquisition_config import DEFAULT_PROFILE_KEY, PROFILES
from database import close_session, create_session, get_session, get_session_summary, init_db
from video_source import open_stream, open_usb_camera, open_video_file


st.set_page_config(
    page_title="visionTar - Controllo qualità",
    page_icon="📷",
    layout="wide",
)


def init_state() -> None:
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    if "profile_key" not in st.session_state:
        st.session_state.profile_key = DEFAULT_PROFILE_KEY
    profile = PROFILES[st.session_state.profile_key]
    defaults = profile.as_percentages()
    for key, value in {
        "roi_left": defaults["left"],
        "roi_right": defaults["right"],
        "roi_top": defaults["top"],
        "roi_bottom": defaults["bottom"],
        "deadline_x": defaults["deadline"],
    }.items():
        st.session_state.setdefault(key, value)


def load_profile(key: str) -> None:
    profile = PROFILES[key]
    values = profile.as_percentages()
    st.session_state.profile_key = key
    st.session_state.roi_left = values["left"]
    st.session_state.roi_right = values["right"]
    st.session_state.roi_top = values["top"]
    st.session_state.roi_bottom = values["bottom"]
    st.session_state.deadline_x = values["deadline"]


def draw_acquisition_overlay(frame, left: int, right: int, top: int, bottom: int, deadline: int, direction: str):
    output = frame.copy()
    height, width = output.shape[:2]

    x1 = int(width * left / 100)
    x2 = int(width * (100 - right) / 100)
    y1 = int(height * top / 100)
    y2 = int(height * (100 - bottom) / 100)
    deadline_x = int(width * deadline / 100)
    deadline_x = max(x1 + 5, min(x2 - 5, deadline_x))

    cv2.rectangle(output, (x1, y1), (x2, y2), (220, 220, 220), 2)
    cv2.line(output, (deadline_x, y1), (deadline_x, y2), (255, 0, 255), 2)

    arrow_y = max(25, y1 - 25)
    if direction == "rtl":
        start = (min(width - 20, x2), arrow_y)
        end = (max(20, x1), arrow_y)
    else:
        start = (max(20, x1), arrow_y)
        end = (min(width - 20, x2), arrow_y)
    cv2.arrowedLine(output, start, end, (255, 255, 255), 2, tipLength=0.08)

    cv2.putText(
        output,
        "ROI",
        (x1 + 8, min(height - 10, y1 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "DEADLINE",
        (max(5, deadline_x - 45), max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def read_uploaded_video_frame(uploaded_file):
    temp_path = None
    try:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.getbuffer())
            temp_path = temp.name

        source = open_video_file(temp_path)
        try:
            ok, frame = source.read()
            if not ok:
                raise RuntimeError("Il video è stato aperto ma non è stato possibile leggere il primo frame.")
            return frame, source.width, source.height, source.fps
        finally:
            source.release()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def read_camera_frame(index: int):
    source = open_usb_camera(index)
    try:
        ok, frame = source.read()
        if not ok:
            raise RuntimeError("La videocamera è stata aperta ma non ha restituito un frame.")
        return frame, source.width, source.height, source.fps
    finally:
        source.release()


def read_stream_frame(url: str):
    source = open_stream(url)
    try:
        ok, frame = source.read()
        if not ok:
            raise RuntimeError("Lo stream è stato aperto ma non ha restituito un frame.")
        return frame, source.width, source.height, source.fps
    finally:
        source.release()


init_db()
init_state()

st.title("visionTar")
st.caption("Controllo qualità visivo - baseline V1")

st.info(
    "Le regolazioni in questa pagina riguardano solo acquisizione, ROI e deadline. "
    "Le soglie di classificazione della baseline V1 restano congelate."
)

st.subheader("Dati produzione")

meta_col1, meta_col2, meta_col3 = st.columns([0.22, 0.48, 0.30])
with meta_col1:
    production_date = st.date_input("Data produzione")
with meta_col2:
    product_description = st.text_input(
        "Descrizione prodotto",
        placeholder="Es. Taralli classici 250 g",
    )
with meta_col3:
    lot_code = st.text_input(
        "Lotto",
        placeholder="Es. L251015A",
    )

left_col, preview_col = st.columns([0.36, 0.64], gap="large")

with left_col:
    st.subheader("Sorgente video")
    source_type = st.radio(
        "Tipo di sorgente",
        ["File video", "Videocamera USB", "Stream RTSP / IP"],
        horizontal=False,
    )

    uploaded = None
    camera_index = 0
    stream_url = ""

    if source_type == "File video":
        uploaded = st.file_uploader("Video di prova", type=["mp4", "avi", "mov", "mkv"])
    elif source_type == "Videocamera USB":
        camera_index = int(st.number_input("Indice videocamera", min_value=0, max_value=10, value=0, step=1))
    else:
        stream_url = st.text_input("URL stream", placeholder="rtsp://...")

    st.subheader("Calibrazione acquisizione")

    profile_labels = {key: profile.label for key, profile in PROFILES.items()}
    selected_key = st.selectbox(
        "Profilo",
        options=list(PROFILES.keys()),
        format_func=lambda key: profile_labels[key],
        index=list(PROFILES.keys()).index(st.session_state.profile_key),
    )

    if st.button("Carica preset", use_container_width=True):
        load_profile(selected_key)
        st.rerun()

    st.slider("ROI sinistra (%)", 0, 45, key="roi_left")
    st.slider("ROI destra (%)", 0, 45, key="roi_right")
    st.slider("ROI alto (%)", 0, 45, key="roi_top")
    st.slider("ROI basso (%)", 0, 45, key="roi_bottom")
    st.slider("Deadline X (%)", 0, 100, key="deadline_x")

    if st.session_state.roi_left + st.session_state.roi_right >= 90:
        st.error("ROI orizzontale non valida: sinistra + destra deve essere inferiore al 90%.")
    if st.session_state.roi_top + st.session_state.roi_bottom >= 90:
        st.error("ROI verticale non valida: alto + basso deve essere inferiore al 90%.")

    profile = PROFILES[st.session_state.profile_key]
    direction_label = "destra → sinistra" if profile.direction == "rtl" else "sinistra → destra"
    st.caption(f"Direzione nastro del preset: {direction_label}")

    st.subheader("Sessione di controllo")

    if st.session_state.active_session_id is None:
        can_start = bool(product_description.strip() and lot_code.strip())
        if st.button(
            "Avvia sessione",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
        ):
            st.session_state.active_session_id = create_session(
                production_date=production_date,
                product_description=product_description,
                lot_code=lot_code,
                profile_key=st.session_state.profile_key,
                source_type=source_type,
            )
            st.rerun()

        if not can_start:
            st.caption("Inserisci descrizione prodotto e lotto per avviare la sessione.")
    else:
        session = get_session(st.session_state.active_session_id)
        if session:
            st.success(
                f"Sessione #{session['id']} attiva - "
                f"{session['product_description']} - lotto {session['lot_code']}"
            )

        if st.button("Chiudi sessione", use_container_width=True):
            close_session(st.session_state.active_session_id)
            st.session_state.last_closed_session_id = st.session_state.active_session_id
            st.session_state.active_session_id = None
            st.rerun()

with preview_col:
    st.subheader("Anteprima acquisizione")

    frame_data = None
    try:
        if source_type == "File video" and uploaded is not None:
            frame_data = read_uploaded_video_frame(uploaded)
        elif source_type == "Videocamera USB" and st.button("Leggi frame dalla videocamera"):
            frame_data = read_camera_frame(camera_index)
        elif source_type == "Stream RTSP / IP" and stream_url and st.button("Leggi frame dallo stream"):
            frame_data = read_stream_frame(stream_url)
    except Exception as exc:
        st.error(str(exc))

    if frame_data is not None:
        frame, width, height, fps = frame_data
        overlay = draw_acquisition_overlay(
            frame,
            st.session_state.roi_left,
            st.session_state.roi_right,
            st.session_state.roi_top,
            st.session_state.roi_bottom,
            st.session_state.deadline_x,
            profile.direction,
        )
        st.image(overlay, channels="RGB", use_container_width=True)
        st.caption(f"Sorgente: {width} × {height} px | FPS rilevati: {fps:.2f}")
    else:
        st.markdown(
            "Seleziona un file oppure leggi un frame dalla videocamera/stream. "
            "La stessa interfaccia verrà usata per il video live durante l'inferenza."
        )

st.divider()
st.subheader("Contatore scarti")

summary_session_id = st.session_state.active_session_id or st.session_state.get("last_closed_session_id")
if summary_session_id is not None:
    summary = get_session_summary(summary_session_id)
    total_col, cotto_col, poco_col, blob_col, rotto_col = st.columns(5)
    total_col.metric("Totale scarti", summary["totale_scarti"])
    cotto_col.metric("Cotto", summary["cotto"])
    poco_col.metric("Poco", summary["poco"])
    blob_col.metric("Blob", summary["blob"])
    rotto_col.metric("Rotto", summary["rotto"])

    summary_session = get_session(summary_session_id)
    if summary_session:
        stato = "chiusa" if summary_session["ended_at"] else "attiva"
        st.caption(
            f"Sessione #{summary_session['id']} {stato} | "
            f"{summary_session['production_date']} | "
            f"{summary_session['product_description']} | "
            f"lotto {summary_session['lot_code']}"
        )
else:
    st.caption("Il contatore sarà disponibile dopo l'avvio della prima sessione.")

st.subheader("Uscita verso il futuro sistema di scarto")
st.write(
    "Vengono registrati solo i pezzi difettosi (cotto, poco, blob, rotto). "
    "Per ogni evento sono previste classe, confidence, coordinate in pixel e normalizzate, "
    "frame e timestamp. Il collegamento fisico a encoder/attuatore resta separato dal Capstone."
)
