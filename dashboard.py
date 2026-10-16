from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from acquisition_config import DEFAULT_BELT_WIDTH_CM, DEFAULT_PROFILE_KEY, DEFAULT_TRAPDOOR_COUNT, PROFILES
from database import (
    add_defect_event,
    build_lot_report_rows,
    build_session_report_rows,
    close_session,
    create_session,
    get_session,
    get_session_summary,
    init_db,
)
from video_source import open_usb_camera, open_video_file
from visionTar import DEFAULT_MODEL_PATH, process_video


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


def draw_acquisition_overlay(
    frame,
    left: int,
    right: int,
    top: int,
    bottom: int,
    deadline: int,
    direction: str,
    trapdoor_count: int,
):
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

    gates = max(1, int(trapdoor_count))
    for gate_idx in range(1, gates):
        gate_y = y1 + int((y2 - y1) * gate_idx / gates)
        cv2.line(output, (x1, gate_y), (x2, gate_y), (150, 150, 150), 1)

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


def build_trapdoor_panel(
    trapdoor_count: int,
    panel_height: int,
    panel_width: int = 56,
    active_gates: list[int] | None = None,
    label: str | None = None,
    track_id: int | None = None,
):
    """Crea una matrice botole con la stessa altezza visiva della ROI video."""
    import numpy as np

    gates = max(1, int(trapdoor_count))
    active = set(active_gates or [])
    panel_h = max(gates, int(panel_height))
    panel_w = max(42, int(panel_width))
    canvas = np.full((panel_h, panel_w, 3), 245, dtype="uint8")

    cell_h = panel_h / gates
    for gate in range(1, gates + 1):
        y1 = int((gate - 1) * cell_h)
        y2 = min(panel_h - 1, int(gate * cell_h) - 1)
        fill = (40, 40, 220) if gate in active else (225, 225, 225)
        border = (20, 20, 20) if gate in active else (145, 145, 145)
        cv2.rectangle(canvas, (0, y1), (panel_w - 1, y2), fill, -1)
        cv2.rectangle(canvas, (0, y1), (panel_w - 1, y2), border, 1)

        cell_pixels = max(1, y2 - y1 + 1)
        if cell_pixels >= 8:
            baseline = y1 + max(7, min(cell_pixels - 1, int(cell_pixels * 0.72)))
            cv2.putText(
                canvas,
                str(gate),
                (4, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.26,
                (255, 255, 255) if gate in active else (65, 65, 65),
                1,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def crop_to_roi(frame, left: int, right: int, top: int, bottom: int):
    height, width = frame.shape[:2]
    x1 = int(width * left / 100)
    x2 = int(width * (100 - right) / 100)
    y1 = int(height * top / 100)
    y2 = int(height * (100 - bottom) / 100)
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    return frame[y1:y2, x1:x2]


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


@st.cache_resource
def load_cached_model():
    return YOLO(str(DEFAULT_MODEL_PATH))


def prepare_uploaded_video(uploaded_file) -> Path:
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if (
        st.session_state.get("prepared_video_signature") == signature
        and st.session_state.get("prepared_video_path")
    ):
        return Path(st.session_state.prepared_video_path)

    work_dir = Path(tempfile.mkdtemp(prefix="visiontar_input_"))
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    input_path = work_dir / f"input{suffix}"
    input_path.write_bytes(uploaded_file.getvalue())

    st.session_state.prepared_video_signature = signature
    st.session_state.prepared_video_path = str(input_path)
    return input_path


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned.strip("_") or "lotto"


init_db()
init_state()

if "display_mode" not in st.session_state:
    st.session_state.display_mode = "calibration"
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None
if "last_video_signature" not in st.session_state:
    st.session_state.last_video_signature = None
if "last_live_frame" not in st.session_state:
    st.session_state.last_live_frame = None
if "trapdoor_count" not in st.session_state:
    st.session_state.trapdoor_count = DEFAULT_TRAPDOOR_COUNT
if "belt_width_cm" not in st.session_state:
    st.session_state.belt_width_cm = DEFAULT_BELT_WIDTH_CM
if "machine_events_bytes" not in st.session_state:
    st.session_state.machine_events_bytes = None
if "machine_events_filename" not in st.session_state:
    st.session_state.machine_events_filename = None
if "camera_preview_frame" not in st.session_state:
    st.session_state.camera_preview_frame = None

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        max-width: 1480px;
    }
    h1, h2, h3, h4 { margin-top: 0.15rem !important; margin-bottom: 0.25rem !important; }
    div[data-testid="stVerticalBlock"] { gap: 0.30rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 10px;
        padding: 0.35rem 0.55rem;
        background: rgba(128,128,128,.04);
    }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stFileUploader"] { margin-bottom: -0.35rem; }
    div[data-testid="stFileUploaderDropzone"] {
        min-height: 2.4rem;
        padding: 0.25rem 0.45rem;
    }
    div[data-testid="stFileUploaderDropzone"] section {
        padding: 0 !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    div[data-baseweb="select"] > div {
        min-height: 2.15rem;
    }
    div[data-testid="stNumberInput"] input {
        min-height: 2.15rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    div[data-testid="stSlider"] {
        padding-top: 0;
        padding-bottom: 0;
    }
    .visiontar-header {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: .01em;
        margin: 0;
    }
    .visiontar-sub {
        opacity: .72;
        font-size: .82rem;
        margin-top: -0.2rem;
    }
    .machine-note {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 9px;
        padding: .42rem .6rem;
        font-size: .80rem;
        line-height: 1.25rem;
    }
    button[kind="primary"] { min-height: 2.55rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([0.72, 0.28])
with header_left:
    st.markdown('<div class="visiontar-header">visionTar</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="visiontar-sub">Controllo qualità visivo · YOLO + OpenCV · simulazione scarto</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.caption("Interfaccia produzione · target 15\"")

meta1, meta2, meta3 = st.columns([0.22, 0.48, 0.30])
with meta1:
    production_date = st.date_input("Data")
with meta2:
    product_description = st.text_input("Prodotto", placeholder="Taralli classici 250 g")
with meta3:
    lot_code = st.text_input("Lotto", placeholder="L251016A")

gate_width_cm = (
    st.session_state.belt_width_cm
    / max(1, st.session_state.trapdoor_count)
)

invalid_roi = (
    st.session_state.roi_left + st.session_state.roi_right >= 90
    or st.session_state.roi_top + st.session_state.roi_bottom >= 90
)
if invalid_roi:
    st.error("Area nastro non valida: riduci i margini.")

profile = PROFILES[st.session_state.profile_key]

input_col, vision_col, right_col = st.columns([0.17, 0.53, 0.30], gap="small")

with input_col:
    st.markdown("#### Input")

    source_mode = st.selectbox(
        "Sorgente",
        ["Video registrato", "Videocamera USB"],
        index=0,
        help="File video oppure videocamera USB.",
    )

    uploaded = None
    camera_index = 0
    input_path = None

    if source_mode == "Video registrato":
        uploaded = st.file_uploader(
            "Video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
    else:
        camera_index = int(
            st.number_input(
                "Camera",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
        )
        if st.button("Leggi camera", use_container_width=True):
            try:
                camera = open_usb_camera(camera_index)
                try:
                    ok, camera_frame = camera.read()
                    if not ok:
                        raise RuntimeError("Nessun fotogramma ricevuto.")
                    st.session_state.camera_preview_frame = camera_frame
                    st.session_state.display_mode = "calibration"
                finally:
                    camera.release()
            except Exception as exc:
                st.error(f"Camera non disponibile: {exc}")

    if uploaded is not None:
        current_signature = f"{uploaded.name}:{uploaded.size}"
        if current_signature != st.session_state.last_video_signature:
            st.session_state.last_video_signature = current_signature
            st.session_state.display_mode = "calibration"
            st.session_state.analysis_summary = None
            st.session_state.last_live_frame = None
            st.session_state.machine_events_bytes = None
            st.session_state.machine_events_filename = None

        try:
            input_path = prepare_uploaded_video(uploaded)
        except Exception as exc:
            st.error(f"Errore caricamento video: {exc}")

    profile_labels = {key: profile.label for key, profile in PROFILES.items()}
    selected_key = st.selectbox(
        "Profilo",
        options=list(PROFILES.keys()),
        format_func=lambda key: profile_labels[key],
        index=list(PROFILES.keys()).index(st.session_state.profile_key),
    )
    if selected_key != st.session_state.profile_key:
        st.session_state.profile_key = selected_key
        profile = PROFILES[selected_key]

with vision_col:
    st.markdown("#### Visione linea")

    frame_data = None
    if source_mode == "Video registrato" and uploaded is not None:
        try:
            frame_data = read_uploaded_video_frame(uploaded)
        except Exception as exc:
            st.error(f"Video non leggibile: {exc}")
    elif (
        source_mode == "Videocamera USB"
        and st.session_state.camera_preview_frame is not None
    ):
        camera_frame = st.session_state.camera_preview_frame
        h, w = camera_frame.shape[:2]
        frame_data = (camera_frame, w, h, 0.0)

    display_video_width = 360
    display_gate_width = 64
    display_video_height = 210
    if frame_data is not None:
        raw_roi = crop_to_roi(
            frame_data[0],
            st.session_state.roi_left,
            st.session_state.roi_right,
            st.session_state.roi_top,
            st.session_state.roi_bottom,
        )
        raw_h, raw_w = raw_roi.shape[:2]
        display_video_height = max(
            st.session_state.trapdoor_count,
            int(round(display_video_width * raw_h / max(1, raw_w))),
        )

    trapdoor_col, video_col = st.columns([0.16, 0.84], gap="small")
    with trapdoor_col:
        trapdoor_frame = st.empty()
        trapdoor_frame.image(
            build_trapdoor_panel(
                st.session_state.trapdoor_count,
                panel_height=display_video_height,
                panel_width=display_gate_width,
            ),
            channels="RGB",
            width=display_gate_width,
        )
    with video_col:
        video_frame = st.empty()
        video_caption = st.empty()
        progress_area = st.empty()

    if frame_data is None:
        if source_mode == "Videocamera USB":
            video_frame.info("Collega/acquisisci la videocamera.")
        else:
            video_frame.info("Carica un video per iniziare.")
    elif st.session_state.display_mode == "calibration":
        frame, width, height, fps = frame_data
        overlay = draw_acquisition_overlay(
            frame,
            st.session_state.roi_left,
            st.session_state.roi_right,
            st.session_state.roi_top,
            st.session_state.roi_bottom,
            st.session_state.deadline_x,
            profile.direction,
            st.session_state.trapdoor_count,
        )
        cropped_overlay = crop_to_roi(
            overlay,
            st.session_state.roi_left,
            st.session_state.roi_right,
            st.session_state.roi_top,
            st.session_state.roi_bottom,
        )
        video_frame.image(
            cropped_overlay,
            channels="RGB",
            width=display_video_width,
        )
        crop_h, crop_w = cropped_overlay.shape[:2]
        video_caption.caption(
            f"AREA NASTRO · {crop_w}×{crop_h} · "
            f"{st.session_state.trapdoor_count} botole"
        )

with right_col:
    st.markdown("#### Sessione")

    can_start = bool(
        source_mode == "Video registrato"
        and product_description.strip()
        and lot_code.strip()
        and input_path is not None
        and not invalid_roi
    )

    start_clicked = st.button(
        "AVVIA SESSIONE",
        type="primary",
        use_container_width=True,
        disabled=not can_start,
    )

    if source_mode == "Videocamera USB":
        st.caption("Live camera predisposto; validazione Capstone eseguita su video registrato.")

    st.markdown("#### Scarti")
    summary_session_id = st.session_state.get("last_closed_session_id")
    if summary_session_id is not None:
        summary = get_session_summary(summary_session_id)
    else:
        summary = {
            "totale_scarti": 0,
            "cotto": 0,
            "poco": 0,
            "blob": 0,
            "rotto": 0,
        }

    m1, m2 = st.columns(2)
    with m1:
        total_metric = st.empty()
    with m2:
        rotto_metric = st.empty()
    m3, m4 = st.columns(2)
    with m3:
        cotto_metric = st.empty()
    with m4:
        poco_metric = st.empty()
    blob_metric = st.empty()

    def render_counters(values: dict[str, int]) -> None:
        total_metric.metric("Totale", values["totale_scarti"])
        rotto_metric.metric("Rotto", values["rotto"])
        cotto_metric.metric("Cotto", values["cotto"])
        poco_metric.metric("Poco", values["poco"])
        blob_metric.metric("Blob", values["blob"])

    render_counters(summary)

    status_box = st.empty()
    if st.session_state.analysis_summary:
        status_box.success(st.session_state.analysis_summary)
    else:
        status_box.caption("Pronto per la sessione.")

    cumulative_rows_ui = build_session_report_rows()
    if cumulative_rows_ui:
        lot_codes_ui = list(dict.fromkeys(row["lot_code"] for row in cumulative_rows_ui))
        st.markdown("#### Esportazione")
        export_mode_ui = st.selectbox(
            "Modalità export",
            ["Seleziona lotto", "Lotto da - a", "Esporta tutti"],
            label_visibility="collapsed",
            key="export_mode_ui",
        )

        filtered_rows_ui = cumulative_rows_ui
        export_suffix_ui = "tutti"

        if export_mode_ui == "Seleziona lotto":
            selected_lot_ui = st.selectbox(
                "Lotto",
                lot_codes_ui,
                key="export_single_lot_ui",
            )
            filtered_rows_ui = [
                row for row in cumulative_rows_ui
                if row["lot_code"] == selected_lot_ui
            ]
            export_suffix_ui = safe_name(selected_lot_ui)

        elif export_mode_ui == "Lotto da - a":
            range_ui_1, range_ui_2 = st.columns(2)
            with range_ui_1:
                lot_from_ui = st.selectbox(
                    "Da",
                    lot_codes_ui,
                    index=0,
                    key="export_lot_from_ui",
                )
            with range_ui_2:
                lot_to_ui = st.selectbox(
                    "A",
                    lot_codes_ui,
                    index=len(lot_codes_ui) - 1,
                    key="export_lot_to_ui",
                )

            start_ui = lot_codes_ui.index(lot_from_ui)
            end_ui = lot_codes_ui.index(lot_to_ui)
            if start_ui > end_ui:
                start_ui, end_ui = end_ui, start_ui
            selected_range_ui = set(lot_codes_ui[start_ui:end_ui + 1])
            filtered_rows_ui = [
                row for row in cumulative_rows_ui
                if row["lot_code"] in selected_range_ui
            ]
            export_suffix_ui = (
                f"{safe_name(lot_codes_ui[start_ui])}_"
                f"{safe_name(lot_codes_ui[end_ui])}"
            )

        export_rows_ui = [
            {
                "Rilevazione": row["rilevazione_id"],
                "Data produzione": row["production_date"],
                "Prodotto": row["product_description"],
                "Lotto": row["lot_code"],
                "Inizio": row["started_at"],
                "Fine": row["ended_at"],
                "Totale scarti": row["totale_scarti"],
                "Cotto": row["cotto"],
                "Poco": row["poco"],
                "Blob": row["blob"],
                "Rotto": row["rotto"],
            }
            for row in filtered_rows_ui
        ]
        export_fields_ui = [
            "Rilevazione",
            "Data produzione",
            "Prodotto",
            "Lotto",
            "Inizio",
            "Fine",
            "Totale scarti",
            "Cotto",
            "Poco",
            "Blob",
            "Rotto",
        ]
        export_output_ui = io.StringIO()
        export_writer_ui = csv.DictWriter(
            export_output_ui,
            fieldnames=export_fields_ui,
            delimiter=";",
            lineterminator="\n",
        )
        export_writer_ui.writeheader()
        export_writer_ui.writerows(export_rows_ui)

        st.download_button(
            "Esporta CSV",
            data=export_output_ui.getvalue().encode("utf-8-sig"),
            file_name=f"report_lotti_{export_suffix_ui}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("#### Area nastro")
belt1, belt2, belt3, belt4, belt5, belt6, belt7, belt8 = st.columns(
    [0.08, 0.09, 0.14, 0.14, 0.14, 0.14, 0.15, 0.12],
    gap="small",
)

with belt1:
    st.number_input(
        "Botole",
        min_value=2,
        max_value=40,
        step=1,
        key="trapdoor_count",
    )
with belt2:
    st.number_input(
        "Nastro cm",
        min_value=10.0,
        max_value=120.0,
        step=1.0,
        key="belt_width_cm",
    )
with belt3:
    st.slider("SX", 0, 45, key="roi_left")
with belt4:
    st.slider("DX", 0, 45, key="roi_right")
with belt5:
    st.slider("ALTO", 0, 45, key="roi_top")
with belt6:
    st.slider("BASSO", 0, 45, key="roi_bottom")
with belt7:
    st.slider("DEADLINE", 0, 100, key="deadline_x")
with belt8:
    st.write("")
    if st.button("Preset", use_container_width=True):
        load_profile(st.session_state.profile_key)
        st.session_state.display_mode = "calibration"
        st.rerun()

gate_width_cm = (
    st.session_state.belt_width_cm
    / max(1, st.session_state.trapdoor_count)
)
st.caption(
    f"{st.session_state.trapdoor_count} botole · "
    f"{st.session_state.belt_width_cm:.0f} cm nastro · "
    f"{gate_width_cm:.1f} cm/botola"
)


if start_clicked:
    if not DEFAULT_MODEL_PATH.exists():
        st.error(f"Modello YOLO non trovato: {DEFAULT_MODEL_PATH}")
    else:
        prep_status = st.status("Preparazione...", expanded=False)
        try:
            model = load_cached_model()
            prep_status.update(label="Modello pronto", state="complete", expanded=False)
        except Exception as exc:
            prep_status.update(label="Errore modello", state="error")
            st.error(f"Impossibile caricare YOLO: {exc}")
            model = None

        if model is not None:
            st.session_state.display_mode = "running"
            st.session_state.analysis_summary = None

            progress = progress_area.progress(0, text="Analisi AI avviata")

            def update_progress(current_frame: int, total_frames: int) -> None:
                if total_frames > 0:
                    progress.progress(
                        min(1.0, current_frame / total_frames),
                        text=f"YOLO + OpenCV · frame {current_frame}/{total_frames}",
                    )

            live_counts = {
                "totale_scarti": 0,
                "cotto": 0,
                "poco": 0,
                "blob": 0,
                "rotto": 0,
            }
            render_counters(live_counts)

            def update_live_counter(defect_class: str, _track_id: int) -> None:
                if defect_class in {"cotto", "poco", "blob", "rotto"}:
                    live_counts[defect_class] += 1
                    live_counts["totale_scarti"] += 1
                    render_counters(live_counts)

            def update_gate_panel(
                active_gates: list[int],
                active_label: str | None,
                active_track_id: int | None,
            ) -> None:
                trapdoor_frame.image(
                    build_trapdoor_panel(
                        st.session_state.trapdoor_count,
                        panel_height=display_video_height,
                        panel_width=display_gate_width,
                        active_gates=active_gates,
                        label=active_label,
                        track_id=active_track_id,
                    ),
                    channels="RGB",
                    width=display_gate_width,
                )

            def update_live_frame(current_frame: int, total_frames: int, frame) -> None:
                cropped = crop_to_roi(
                    frame,
                    st.session_state.roi_left,
                    st.session_state.roi_right,
                    st.session_state.roi_top,
                    st.session_state.roi_bottom,
                )
                height, width = cropped.shape[:2]
                target_width = display_video_width
                if width > target_width:
                    scale = target_width / width
                    preview = cv2.resize(
                        cropped,
                        (target_width, max(1, int(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    preview = cropped

                ok, encoded = cv2.imencode(
                    ".jpg",
                    preview,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 76],
                )
                if ok:
                    video_frame.image(
                        encoded.tobytes(),
                        width=display_video_width,
                    )

                if total_frames > 0:
                    video_caption.caption(
                        f"IN ANALISI · frame {current_frame}/{total_frames} · "
                        f"OpenCV simula {st.session_state.trapdoor_count} botole"
                    )

            run_input = Path(input_path)

            try:
                detections_path = process_video(
                    video_path=run_input,
                    model_path=DEFAULT_MODEL_PATH,
                    test_mode=True,
                    roi_left=st.session_state.roi_left / 100.0,
                    roi_right=st.session_state.roi_right / 100.0,
                    roi_top=st.session_state.roi_top / 100.0,
                    roi_bottom=st.session_state.roi_bottom / 100.0,
                    deadline_ratio=st.session_state.deadline_x / 100.0,
                    show_window=False,
                    annotated_output_path=None,
                    progress_callback=update_progress,
                    frame_callback=update_live_frame,
                    gate_callback=update_gate_panel,
                    defect_callback=update_live_counter,
                    model_instance=model,
                    trapdoor_count=st.session_state.trapdoor_count,
                )

                scarti_path = run_input.with_name(
                    f"{run_input.stem}_scarti_test.csv"
                )
                finali_path = run_input.with_name(
                    f"{run_input.stem}_finali_test.csv"
                )

                # Diagnostica non invasiva: individua track con evidenza YOLO
                # "poco" che non arrivano al responso finale POCO. Non cambia
                # alcuna soglia del classificatore.
                poco_track_stats: dict[int, dict[str, float]] = {}
                if Path(detections_path).exists():
                    with Path(detections_path).open(
                        "r",
                        encoding="utf-8-sig",
                        newline="",
                    ) as detection_file:
                        for row in csv.DictReader(detection_file):
                            if (row.get("classe") or "").strip() != "poco":
                                continue
                            raw_track = (row.get("track_id") or "").strip()
                            if not raw_track:
                                continue
                            track_id = int(float(raw_track))
                            try:
                                v_value = float(row.get("v_median") or 0.0)
                                s_value = float(row.get("s_median") or 0.0)
                                lab_l_value = float(row.get("lab_l_median") or 0.0)
                                confidence_value = float(row.get("confidence") or 0.0)
                            except ValueError:
                                continue

                            if not (
                                v_value >= 212.0
                                and lab_l_value >= 208.0
                                and s_value <= 105.0
                            ):
                                continue

                            stats = poco_track_stats.setdefault(
                                track_id,
                                {"count": 0.0, "confidence_sum": 0.0},
                            )
                            stats["count"] += 1.0
                            stats["confidence_sum"] += confidence_value

                final_by_track: dict[int, dict[str, str]] = {}
                if finali_path.exists():
                    with finali_path.open(
                        "r",
                        encoding="utf-8-sig",
                        newline="",
                    ) as final_file:
                        for row in csv.DictReader(final_file):
                            raw_track = (row.get("track_id") or "").strip()
                            if raw_track:
                                final_by_track[int(float(raw_track))] = row

                poco_candidates_lost = []
                for track_id, stats in poco_track_stats.items():
                    if stats["count"] < 2:
                        continue
                    final_row = final_by_track.get(track_id, {})
                    final_label = (final_row.get("responso_finale") or "").strip()
                    if final_label == "poco":
                        continue
                    poco_candidates_lost.append(
                        {
                            "track_id": track_id,
                            "osservazioni": int(stats["count"]),
                            "conf_media": (
                                stats["confidence_sum"] / stats["count"]
                                if stats["count"] > 0
                                else 0.0
                            ),
                            "finale": final_label or "nessuno",
                            "supporto_colore": (
                                final_row.get("supporto_colore") or ""
                            ).strip(),
                            "lab_l": (
                                final_row.get("track_lab_l_median") or ""
                            ).strip(),
                        }
                    )

                st.session_state.poco_diagnostic = poco_candidates_lost

                session_id = create_session(
                    production_date=production_date,
                    product_description=product_description,
                    lot_code=lot_code,
                    profile_key=st.session_state.profile_key,
                    source_type="File video",
                )

                imported = 0
                if scarti_path.exists():
                    st.session_state.machine_events_bytes = scarti_path.read_bytes()
                    st.session_state.machine_events_filename = (
                        f"eventi_scarto_{safe_name(lot_code)}_"
                        f"{safe_name(product_description)}.csv"
                    )
                    with scarti_path.open(
                        "r",
                        encoding="utf-8-sig",
                        newline="",
                    ) as csv_file:
                        reader = csv.DictReader(csv_file)
                        for row in reader:
                            classe = (row.get("classe") or "").strip()
                            if classe not in {"cotto", "poco", "blob", "rotto"}:
                                continue

                            inserted = add_defect_event(
                                session_id,
                                classe,
                                track_id=int(row["track_id"]) if row.get("track_id") else None,
                                confidence=float(row["confidence"]) if row.get("confidence") else None,
                                x_px=int(float(row["x_px"])) if row.get("x_px") else None,
                                y_px=int(float(row["y_px"])) if row.get("y_px") else None,
                                x_norm=float(row["x_norm"]) if row.get("x_norm") else None,
                                y_norm=float(row["y_norm"]) if row.get("y_norm") else None,
                                frame_finale=int(float(row["frame_finale"])) if row.get("frame_finale") else None,
                                timestamp_s=float(row["timestamp_s"]) if row.get("timestamp_s") else None,
                            )
                            if inserted:
                                imported += 1

                close_session(session_id)

                cumulative_rows = build_session_report_rows()
                export_rows = [
                    {
                        "Rilevazione": row["rilevazione_id"],
                        "Data produzione": row["production_date"],
                        "Prodotto": row["product_description"],
                        "Lotto": row["lot_code"],
                        "Inizio": row["started_at"],
                        "Fine": row["ended_at"],
                        "Totale scarti": row["totale_scarti"],
                        "Cotto": row["cotto"],
                        "Poco": row["poco"],
                        "Blob": row["blob"],
                        "Rotto": row["rotto"],
                    }
                    for row in cumulative_rows
                ]
                cumulative_fields = [
                    "Rilevazione",
                    "Data produzione",
                    "Prodotto",
                    "Lotto",
                    "Inizio",
                    "Fine",
                    "Totale scarti",
                    "Cotto",
                    "Poco",
                    "Blob",
                    "Rotto",
                ]
                cumulative_output = io.StringIO()
                cumulative_writer = csv.DictWriter(
                    cumulative_output,
                    fieldnames=cumulative_fields,
                    delimiter=";",
                    lineterminator="\n",
                )
                cumulative_writer.writeheader()
                cumulative_writer.writerows(export_rows)
                Path("report_lotti.csv").write_text(
                    cumulative_output.getvalue(),
                    encoding="utf-8-sig",
                )

                st.session_state.last_closed_session_id = session_id
                poco_diag_count = len(
                    st.session_state.get("poco_diagnostic", [])
                )
                diagnostic_suffix = (
                    f" · {poco_diag_count} track POCO forte non confermata"
                    if poco_diag_count
                    else ""
                )
                st.session_state.analysis_summary = (
                    f"Sessione #{session_id} · {imported} scarti"
                    f"{diagnostic_suffix}"
                )
                st.session_state.display_mode = "processed"
                progress.progress(1.0, text="Analisi completata")
                st.rerun()

            except Exception as exc:
                progress_area.empty()
                st.session_state.display_mode = "calibration"
                st.error(f"Errore durante l'analisi: {exc}")

st.markdown(
    f"""
    <div class="machine-note">
    <b>Simulatore OpenCV:</b> nastro {st.session_state.belt_width_cm:.0f} cm ·
    {st.session_state.trapdoor_count} botole ·
    {gate_width_cm:.1f} cm/botola. Le botole evidenziate nel video rappresentano
    il comando candidato; il timing encoder resta simulato fino al collegamento hardware.
    </div>
    """,
    unsafe_allow_html=True,
)

