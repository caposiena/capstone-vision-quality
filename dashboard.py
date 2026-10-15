from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from acquisition_config import DEFAULT_PROFILE_KEY, DEFAULT_TRAPDOOR_COUNT, PROFILES
from database import (
    add_defect_event,
    build_lot_report_rows,
    close_session,
    create_session,
    get_session,
    get_session_summary,
    init_db,
)
from video_source import open_video_file
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
if "machine_events_bytes" not in st.session_state:
    st.session_state.machine_events_bytes = None
if "machine_events_filename" not in st.session_state:
    st.session_state.machine_events_filename = None

st.title("visionTar")
st.caption("Controllo qualità visivo - baseline V1")

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

st.divider()

settings_col, viewer_col = st.columns([0.34, 0.66], gap="large")

with settings_col:
    st.markdown("#### Impostazioni video")

    uploaded = st.file_uploader(
        "Seleziona video di prova",
        type=["mp4", "avi", "mov", "mkv"],
    )

    input_path = None
    if uploaded is not None:
        current_signature = f"{uploaded.name}:{uploaded.size}"
        if current_signature != st.session_state.last_video_signature:
            st.session_state.last_video_signature = current_signature
            st.session_state.display_mode = "calibration"
            st.session_state.analysis_summary = None
            st.session_state.last_live_frame = None

        try:
            input_path = prepare_uploaded_video(uploaded)
            st.caption("Video pronto per l'analisi.")
        except Exception as exc:
            st.error(f"Errore nel caricamento locale del video: {exc}")

    profile_labels = {key: profile.label for key, profile in PROFILES.items()}
    selected_key = st.selectbox(
        "Profilo acquisizione",
        options=list(PROFILES.keys()),
        format_func=lambda key: profile_labels[key],
        index=list(PROFILES.keys()).index(st.session_state.profile_key),
    )

    if st.button("Applica preset", use_container_width=True):
        load_profile(selected_key)
        st.session_state.display_mode = "calibration"
        st.rerun()

    st.slider("ROI sinistra (%)", 0, 45, key="roi_left")
    st.slider("ROI destra (%)", 0, 45, key="roi_right")
    st.slider("ROI alto (%)", 0, 45, key="roi_top")
    st.slider("ROI basso (%)", 0, 45, key="roi_bottom")
    st.slider("Deadline X (%)", 0, 100, key="deadline_x")
    st.number_input(
        "Numero botole trasversali",
        min_value=2,
        max_value=32,
        step=1,
        key="trapdoor_count",
        help="Suddivide la larghezza utile del nastro in zone di scarto.",
    )

    invalid_roi = (
        st.session_state.roi_left + st.session_state.roi_right >= 90
        or st.session_state.roi_top + st.session_state.roi_bottom >= 90
    )
    if invalid_roi:
        st.error("ROI non valida: riduci i margini.")

    profile = PROFILES[st.session_state.profile_key]
    direction_label = (
        "destra → sinistra" if profile.direction == "rtl" else "sinistra → destra"
    )
    st.caption(f"{profile.label} · {direction_label}")

with viewer_col:
    st.markdown("#### Schermata video")

    frame_data = None
    if uploaded is not None:
        try:
            frame_data = read_uploaded_video_frame(uploaded)
        except Exception as exc:
            st.error(f"Impossibile leggere il video: {exc}")

    control_col1, control_col2, control_spacer = st.columns([0.22, 0.30, 0.48])

    with control_col1:
        if st.button("Calibra", use_container_width=True):
            st.session_state.display_mode = "calibration"
            st.session_state.last_live_frame = None

    with control_col2:
        can_start = bool(
            product_description.strip()
            and lot_code.strip()
            and input_path is not None
            and not invalid_roi
        )
        start_clicked = st.button(
            "Avvia sessione",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
        )

    screen_left, screen_center, screen_right = st.columns([0.31, 0.38, 0.31])

    with screen_center:
        live_image = st.empty()
        live_caption = st.empty()
        progress_area = st.empty()

        if uploaded is None:
            live_image.info("Carica un video di prova.")
        elif frame_data is not None and st.session_state.display_mode == "calibration":
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
            live_image.image(overlay, channels="RGB", width=340)
            live_caption.caption(
                f"Fotogramma di calibrazione · {width} × {height} px · {fps:.2f} FPS"
            )
        elif st.session_state.last_live_frame is not None:
            live_image.image(
                st.session_state.last_live_frame,
                channels="RGB",
                width=340,
            )

    if start_clicked:
        if not DEFAULT_MODEL_PATH.exists():
            st.error(f"Modello YOLO non trovato: {DEFAULT_MODEL_PATH}")
        else:
            prep_status = st.status("Preparazione sessione...", expanded=False)
            prep_status.write("Video già caricato. Preparo il modello YOLO.")
            try:
                model = load_cached_model()
                prep_status.update(
                    label="Preparazione completata",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                prep_status.update(label="Errore preparazione", state="error")
                st.error(f"Impossibile caricare il modello YOLO: {exc}")
                model = None

            if model is not None:
                st.session_state.display_mode = "running"
                st.session_state.analysis_summary = None

                progress = progress_area.progress(
                    0,
                    text="Analisi AI avviata...",
                )

                def update_progress(current_frame: int, total_frames: int) -> None:
                    if total_frames > 0:
                        progress.progress(
                            min(1.0, current_frame / total_frames),
                            text=f"Analisi AI · frame {current_frame} / {total_frames}",
                        )

                def update_live_frame(current_frame: int, total_frames: int, frame) -> None:
                    height, width = frame.shape[:2]
                    target_width = 340
                    if width > target_width:
                        scale = target_width / width
                        preview = cv2.resize(
                            frame,
                            (target_width, max(1, int(height * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        preview = frame

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        preview,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 72],
                    )
                    if ok:
                        live_image.image(encoded.tobytes())

                    if total_frames > 0:
                        live_caption.caption(
                            f"YOLO al lavoro · frame {current_frame}/{total_frames}"
                        )
                    else:
                        live_caption.caption(
                            f"YOLO al lavoro · frame {current_frame}"
                        )

                run_input = Path(input_path)

                try:
                    process_video(
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
                        model_instance=model,
                        trapdoor_count=st.session_state.trapdoor_count,
                    )

                    scarti_path = run_input.with_name(
                        f"{run_input.stem}_scarti_test.csv"
                    )

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
                    st.session_state.last_closed_session_id = session_id
                    st.session_state.analysis_summary = (
                        f"Sessione #{session_id} completata · {imported} scarti registrati"
                    )
                    st.session_state.display_mode = "processed"
                    progress.progress(1.0, text="Analisi completata")
                    st.rerun()

                except Exception as exc:
                    progress_area.empty()
                    st.session_state.display_mode = "calibration"
                    st.error(f"Errore durante l'analisi YOLO: {exc}")

st.divider()

summary_session_id = st.session_state.get("last_closed_session_id")
st.markdown("#### Contatore scarti")

if summary_session_id is not None:
    summary = get_session_summary(summary_session_id)
    total_col, cotto_col, poco_col, blob_col, rotto_col = st.columns(5)
    total_col.metric("Totale", summary["totale_scarti"])
    cotto_col.metric("Cotto", summary["cotto"])
    poco_col.metric("Poco", summary["poco"])
    blob_col.metric("Blob", summary["blob"])
    rotto_col.metric("Rotto", summary["rotto"])
else:
    st.caption("Il conteggio comparirà al termine della prima sessione.")

st.divider()

status_col, report_col = st.columns([0.50, 0.50], gap="large")

with status_col:
    st.markdown("#### Stato analisi")
    if st.session_state.analysis_summary:
        st.success(st.session_state.analysis_summary)
    else:
        st.caption(
            "Calibra il video e premi Avvia sessione. "
            "Durante l'analisi vedrai YOLO lavorare sul video."
        )

with report_col:
    st.markdown("#### Report lotto")

    if summary_session_id is not None:
        session = get_session(summary_session_id)
        summary = get_session_summary(summary_session_id)

        output = io.StringIO()
        fieldnames = [
            "production_date",
            "product_description",
            "lot_code",
            "totale_scarti",
            "cotto",
            "poco",
            "blob",
            "rotto",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "production_date": session["production_date"],
                "product_description": session["product_description"],
                "lot_code": session["lot_code"],
                "totale_scarti": summary["totale_scarti"],
                "cotto": summary["cotto"],
                "poco": summary["poco"],
                "blob": summary["blob"],
                "rotto": summary["rotto"],
            }
        )

        lot_file_name = (
            f"lotto_{safe_name(session['lot_code'])}_"
            f"{safe_name(session['product_description'])}.csv"
        )

        st.download_button(
            f"Scarica {lot_file_name}",
            data=output.getvalue().encode("utf-8-sig"),
            file_name=lot_file_name,
            mime="text/csv",
            use_container_width=True,
        )

        if (
            st.session_state.machine_events_bytes
            and st.session_state.machine_events_filename
        ):
            st.download_button(
                f"Scarica {st.session_state.machine_events_filename}",
                data=st.session_state.machine_events_bytes,
                file_name=st.session_state.machine_events_filename,
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.caption("Il report sarà disponibile al termine della sessione.")
