from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from mathvse.pipeline.image_inference import infer_image
from mathvse.pipeline.orchestrator import solve_text_expression
from mathvse.recognition.classifier import TorchSymbolClassifier
from mathvse.vision.input import load_image_file


@st.cache_data(show_spinner=False)
def evaluate_expression(expression: str):
    return solve_text_expression(expression)


def evaluate_uploaded_image(
    file_path: str | Path,
    *,
    model_path: str | None = None,
    confidence_threshold: float = 0.55,
    ambiguity_margin: float = 0.08,
):
    path = Path(file_path)
    loaded = load_image_file(path)
    inference = infer_image(
        loaded.image,
        classifier=TorchSymbolClassifier(model_path=model_path),
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        ambiguity_margin=ambiguity_margin,
        loaded_image=loaded,
    )

    return {
        "loaded": loaded,
        "processed": inference.preprocessing,
        "recognition": inference.recognition,
        "inference": inference,
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
        upload_bytes = uploaded.getvalue()
        upload_key = hashlib.sha256(upload_bytes).hexdigest()[:16]
        if st.session_state.get("last_upload_key") != upload_key:
            st.session_state["image_result"] = None
            st.session_state["last_upload_key"] = upload_key
        temp_path = Path("/tmp") / f"mathvse_{upload_key}_{Path(uploaded.name).name}"
        temp_path.write_bytes(upload_bytes)
        result = evaluate_uploaded_image(
            temp_path,
            model_path=st.session_state.get("ocr_model_path") or None,
            confidence_threshold=st.session_state.get("ocr_confidence_threshold", 0.55),
            ambiguity_margin=st.session_state.get("ocr_ambiguity_margin", 0.08),
        )
    except Exception as exc:  # pragma: no cover - UI boundary handling
        st.error(f"Image could not be processed: {exc}")
        return

    loaded = result["loaded"]
    processed = result["processed"]
    recognition = result["recognition"]
    inference = result["inference"]

    st.image(np.asarray(loaded.image[:, :, ::-1]), caption="Original image", use_container_width=True)
    st.caption(f"Image metadata: {loaded.metadata.width}x{loaded.metadata.height}, {loaded.metadata.file_format}, {loaded.metadata.file_size_bytes} bytes")

    if processed is None:
        st.error("Image preprocessing failed; no crops or expression were generated.")
        for warning in inference.warnings:
            st.caption(warning)
        return

    option = st.selectbox("Preprocessing variant", list(processed.variants.keys()))
    preview = _prepare_variant_image(option, processed)
    st.image(preview, caption=f"Variant: {option}", use_container_width=True)

    if processed and processed.bounding_boxes:
        preview_with_boxes = _draw_boxes(preview, processed.bounding_boxes)
        st.image(preview_with_boxes, caption="Foreground component candidates", use_container_width=True)

    if inference.crops:
        st.write("### Component crops and predictions")
        for crop in inference.crops:
            prediction = next((item for item in inference.predictions if item.component_id == crop.component_id), None)
            if crop.image is not None:
                st.image(crop.image, caption=f"{crop.component_id} · bbox {crop.bbox}", width=120)
            if prediction is None:
                st.caption(f"{crop.component_id}: {crop.status}" + (f" · {crop.warning}" if crop.warning else ""))
                continue
            ranked = ", ".join(
                f"{candidate.label}: {candidate.confidence:.3f}"
                for candidate in prediction.alternatives
            )
            st.caption(
                f"{crop.component_id}: {prediction.label} ({prediction.confidence:.3f}) · "
                f"{prediction.status} · candidates: {ranked or 'none'}"
            )

    if inference.status == "model_unavailable":
        st.warning("OCR model or PyTorch is unavailable. No symbol predictions were generated; manual expression solving remains available.")
    elif inference.status != "success":
        st.warning(f"Image recognition status: {inference.status}. Review the component crops or enter a corrected expression.")
    for warning in inference.warnings:
        st.caption(warning)

    st.text_input("Machine expression (unconfirmed)", value=inference.normalized_expression, disabled=True)
    confirmation_key = f"confirmed_expression_{upload_key}"
    reviewed_key = f"ocr_reviewed_{upload_key}"
    if confirmation_key not in st.session_state:
        st.session_state[confirmation_key] = inference.normalized_expression
    corrected_expression = st.text_input("Confirmed / corrected expression", key=confirmation_key)
    reviewed = st.checkbox("I reviewed and confirm this expression", key=reviewed_key)
    if st.button("Solve confirmed expression", disabled=not reviewed or not corrected_expression.strip()):
        st.session_state["image_result"] = evaluate_expression(corrected_expression)

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
            help="Optional path to a compatible local Phase 5A CNN checkpoint.",
        )
        st.session_state["ocr_confidence_threshold"] = st.slider(
            "Minimum symbol confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("ocr_confidence_threshold", 0.55)),
            step=0.05,
        )
        st.session_state["ocr_ambiguity_margin"] = st.slider(
            "Minimum top-class separation",
            min_value=0.0,
            max_value=0.5,
            value=float(st.session_state.get("ocr_ambiguity_margin", 0.08)),
            step=0.01,
        )
        st.caption("Image predictions require manual review before solving.")

    tab_image, tab_manual = st.tabs(["Image input", "Manual expression"])
    with tab_image:
        _render_image_solver()
    with tab_manual:
        _render_manual_solver()


if __name__ == "__main__":
    main()
