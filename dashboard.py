from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from acquisition_config import DEFAULT_PROFILE_KEY, PROFILES
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


init_db()
init_state()

if "display_mode" not in st.session_state:
    st.session_state.display_mode = "calibration"
if "processed_video_bytes" not in st.session_state:
    st.session_state.processed_video_bytes = None
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None

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

    source_type = "File video"
    uploaded = st.file_uploader(
        "Seleziona video di prova",
        type=["mp4", "avi", "mov", "mkv"],
    )

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

    invalid_roi = (
        st.session_state.roi_left + st.session_state.roi_right >= 90
        or st.session_state.roi_top + st.session_state.roi_bottom >= 90
    )
    if invalid_roi:
        st.error("ROI non valida: riduci i margini.")

    profile = PROFILES[st.session_state.profile_key]
    direction_label = "destra → sinistra" if profile.direction == "rtl" else "sinistra → destra"
    st.caption(f"{profile.label} · {direction_label}")

with viewer_col:
    st.markdown("#### Schermata video")

    frame_data = None
    if uploaded is not None:
        try:
            frame_data = read_uploaded_video_frame(uploaded)
        except Exception as exc:
            st.error(f"Impossibile leggere il video: {exc}")

    view_btn1, view_btn2, view_spacer = st.columns([0.24, 0.30, 0.46])
    with view_btn1:
        if st.button("Calibra", use_container_width=True):
            st.session_state.display_mode = "calibration"
            st.session_state.processed_video_bytes = None
    with view_btn2:
        analyze_disabled = (
            uploaded is None
            or st.session_state.active_session_id is None
            or invalid_roi
        )
        analyze_clicked = st.button(
            "Avvia analisi YOLO",
            type="primary",
            use_container_width=True,
            disabled=analyze_disabled,
        )

    if analyze_clicked:
        if not DEFAULT_MODEL_PATH.exists():
            st.error(f"Modello YOLO non trovato: {DEFAULT_MODEL_PATH}")
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="visiontar_dashboard_"))
            input_path = work_dir / "input.mp4"
            annotated_path = work_dir / "elaborato.mp4"
            input_path.write_bytes(uploaded.getvalue())

            with st.spinner("Analisi YOLO in corso..."):
                try:
                    process_video(
                        video_path=input_path,
                        model_path=DEFAULT_MODEL_PATH,
                        test_mode=True,
                        roi_left=st.session_state.roi_left / 100.0,
                        roi_right=st.session_state.roi_right / 100.0,
                        roi_top=st.session_state.roi_top / 100.0,
                        roi_bottom=st.session_state.roi_bottom / 100.0,
                        deadline_ratio=st.session_state.deadline_x / 100.0,
                        show_window=False,
                        annotated_output_path=annotated_path,
                    )

                    scarti_path = input_path.with_name(f"{input_path.stem}_scarti_test.csv")
                    imported = 0
                    if scarti_path.exists():
                        with scarti_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                            reader = csv.DictReader(csv_file)
                            for row in reader:
                                classe = (row.get("classe") or "").strip()
                                if classe not in {"cotto", "poco", "blob", "rotto"}:
                                    continue
                                inserted = add_defect_event(
                                    st.session_state.active_session_id,
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

                    st.session_state.processed_video_bytes = (
                        annotated_path.read_bytes() if annotated_path.exists() else None
                    )
                    st.session_state.analysis_summary = f"Analisi completata · {imported} scarti registrati"
                    st.session_state.display_mode = "processed"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Errore durante l'analisi YOLO: {exc}")

    screen_left, screen_center, screen_right = st.columns([0.20, 0.60, 0.20])

    with screen_center:
        if uploaded is None:
            st.info("Carica un video di prova.")
        elif st.session_state.display_mode == "processed" and st.session_state.processed_video_bytes:
            st.video(st.session_state.processed_video_bytes, width=420)
            if st.session_state.analysis_summary:
                st.success(st.session_state.analysis_summary)
        elif frame_data is not None:
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
            st.image(overlay, channels="RGB", width=420)
            st.caption(
                f"Fotogramma di calibrazione · {width} × {height} px · {fps:.2f} FPS"
            )

st.divider()

session_col, counters_col = st.columns([0.34, 0.66], gap="large")

with session_col:
    st.markdown("#### Sessione")

    if st.session_state.active_session_id is None:
        can_start = bool(
            product_description.strip()
            and lot_code.strip()
            and uploaded is not None
            and not invalid_roi
        )

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
            st.caption("Inserisci prodotto, lotto, video e una ROI valida.")
    else:
        session = get_session(st.session_state.active_session_id)
        if session:
            st.success(
                f"Sessione #{session['id']} · {session['product_description']} · "
                f"lotto {session['lot_code']}"
            )

        if st.button("Chiudi sessione", use_container_width=True):
            close_session(st.session_state.active_session_id)
            st.session_state.last_closed_session_id = st.session_state.active_session_id
            st.session_state.active_session_id = None
            st.rerun()

with counters_col:
    st.markdown("#### Contatore scarti")

    summary_session_id = (
        st.session_state.active_session_id
        or st.session_state.get("last_closed_session_id")
    )

    if summary_session_id is not None:
        summary = get_session_summary(summary_session_id)
        total_col, cotto_col, poco_col, blob_col, rotto_col = st.columns(5)
        total_col.metric("Totale", summary["totale_scarti"])
        cotto_col.metric("Cotto", summary["cotto"])
        poco_col.metric("Poco", summary["poco"])
        blob_col.metric("Blob", summary["blob"])
        rotto_col.metric("Rotto", summary["rotto"])
    else:
        st.caption("Il conteggio comparirà dopo l'avvio della sessione.")

st.divider()

import_col, report_col = st.columns([0.50, 0.50], gap="large")

with import_col:
    st.markdown("#### Stato analisi")
    if st.session_state.analysis_summary:
        st.success(st.session_state.analysis_summary)
    else:
        st.caption("Calibra il video e avvia l'analisi YOLO. I risultati vengono registrati automaticamente.")

with report_col:
    st.markdown("#### Report per lotto")

    report_rows = build_lot_report_rows()
    if report_rows:
        output = io.StringIO()
        fieldnames = [
            "production_date",
            "product_description",
            "lot_code",
            "sessions",
            "totale_scarti",
            "cotto",
            "poco",
            "blob",
            "rotto",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

        st.download_button(
            "Scarica report_lotti.csv",
            data=output.getvalue().encode("utf-8-sig"),
            file_name="report_lotti.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Il report sarà disponibile dopo il primo lotto.")
