from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from mathvse.contracts.recognition import RecognitionResult, SymbolPrediction
from mathvse.pipeline.orchestrator import solve_text_expression
from mathvse.recognition.classifier import BaselineSymbolClassifier
from mathvse.spatial.parser import parse_symbols
from mathvse.vision.input import load_image_file
from mathvse.vision.preprocessing import preprocess_image


@st.cache_data(show_spinner=False)
def evaluate_expression(expression: str):
    return solve_text_expression(expression)


@st.cache_data(show_spinner=False)
def evaluate_uploaded_image(file_path: str | Path, *, model_path: str | None = None):
    path = Path(file_path)
    loaded = load_image_file(path)
    processed = preprocess_image(loaded.image, strategy_name="balanced")

    if model_path:
        classifier = BaselineSymbolClassifier(model_path=model_path)
        status = "ready" if classifier.is_available else "model_unavailable"
        recognition = RecognitionResult(
            status="model_unavailable" if not classifier.is_available else "success",
            symbols=[],
            normalized_text="",
            warnings=["No trained OCR weights were loaded for this environment."] if not classifier.is_available else [],
            model_path=model_path,
            model_status=status,
        )
    else:
        recognition = RecognitionResult(
            status="model_unavailable",
            symbols=[],
            normalized_text="",
            warnings=["No OCR model path configured; image recognition is unavailable."],
            model_path=None,
            model_status="model_unavailable",
        )

    return {
        "loaded": loaded,
        "processed": processed,
        "recognition": recognition,
    }


def _prepare_variant_image(variant_name: str, processed: Any) -> np.ndarray:
    variant = processed.variants.get(variant_name, processed.variants.get("grayscale"))
    if variant is None:
        raise ValueError(f"Unsupported preprocessing variant: {variant_name}")
    if variant.ndim == 2:
        return cv2.cvtColor(variant.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(variant.astype(np.uint8), cv2.COLOR_BGR2RGB)


def _draw_boxes(image_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    if not boxes:
        return image_rgb.copy()
    vis = image_rgb.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def _render_manual_solver():
    st.subheader("Manual expression input")
    default_value = st.session_state.get("manual_expression", "2*x + 5 = 15")
    expression = st.text_input("Equation", value=default_value, key="manual_expression")

    if st.button("Solve expression"):
        st.session_state["manual_result"] = evaluate_expression(expression)

    result = st.session_state.get("manual_result")
    if result is not None:
        st.markdown(f"**Status:** {result.status}")
        if result.solution:
            st.markdown(f"**Solution:** {result.solution}")
        if result.message:
            st.caption(result.message)
        if result.trace:
            st.write("### Transformation steps")
            for step in result.trace:
                st.markdown(f"- **{step.rule_id}**: {step.before} -> {step.after}")
        if result.trace:
            st.write("### Indonesian explanation")
            st.write(result.message)


def _render_image_solver():
    st.subheader("Image input")
    uploaded = st.file_uploader("Upload PNG / JPG / JPEG", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Upload a supported image to validate preprocessing and inspect candidate regions.")
        return

    try:
        temp_path = Path("/tmp") / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        result = evaluate_uploaded_image(temp_path, model_path=st.session_state.get("ocr_model_path"))
    except Exception as exc:  # pragma: no cover - UI boundary handling
        st.error(f"Image could not be processed: {exc}")
        return

    loaded = result["loaded"]
    processed = result["processed"]
    recognition = result["recognition"]

    st.image(np.asarray(loaded.image[:, :, ::-1]), caption="Original image", use_container_width=True)

    option = st.selectbox("Preprocessing variant", list(processed.variants.keys()))
    preview = _prepare_variant_image(option, processed)
    st.image(preview, caption=f"Variant: {option}", use_container_width=True)

    if processed.bounding_boxes:
        preview_with_boxes = _draw_boxes(preview, processed.bounding_boxes)
        st.image(preview_with_boxes, caption="Preliminary candidate boxes", use_container_width=True)

    st.caption(f"Image metadata: {loaded.metadata.width}x{loaded.metadata.height}, {loaded.metadata.file_format}, {loaded.metadata.file_size_bytes} bytes")

    if recognition.model_status == "model_unavailable":
        st.warning("OCR model unavailable. The system cannot fabricate recognized symbols. Use manual expression input or provide a valid model path.")
        st.session_state["recognized_expression"] = ""
    else:
        st.session_state["recognized_expression"] = recognition.normalized_text or ""

    st.text_input("Recognized expression (machine)", value=st.session_state.get("recognized_expression", ""), key="recognized_expression")
    edited_expression = st.text_input("Confirmed / corrected expression", value=st.session_state.get("recognized_expression", "") or "")
    if st.button("Solve confirmed expression"):
        st.session_state["image_result"] = evaluate_expression(edited_expression)

    image_result = st.session_state.get("image_result")
    if image_result is not None:
        st.markdown(f"**Status:** {image_result.status}")
        if image_result.solution:
            st.markdown(f"**Solution:** {image_result.solution}")
        if image_result.message:
            st.caption(image_result.message)
        if image_result.trace:
            st.write("### Transformation steps")
            for step in image_result.trace:
                st.markdown(f"- **{step.rule_id}**: {step.before} -> {step.after}")


def main() -> None:
    st.set_page_config(page_title="MathVSE", page_icon="∑", layout="wide")
    st.title("MathVSE")
    st.caption("Local-first mathematical equation solver and explainer")
    st.text("Processing stays local and offline; no cloud OCR or remote APIs are used.")

    st.session_state.setdefault("ocr_model_path", "")
    st.session_state.setdefault("manual_result", None)
    st.session_state.setdefault("image_result", None)

    with st.sidebar:
        st.header("Settings")
        st.session_state["ocr_model_path"] = st.text_input(
            "OCR model path",
            value=st.session_state.get("ocr_model_path", ""),
            help="Optional local model path for a future offline OCR model. Missing weights are reported explicitly.",
        )
        st.caption("The default OCR path is intentionally absent; no fake predictions are produced.")

    tab_image, tab_manual = st.tabs(["Image input", "Manual expression"])
    with tab_image:
        _render_image_solver()
    with tab_manual:
        _render_manual_solver()


if __name__ == "__main__":
    main()
