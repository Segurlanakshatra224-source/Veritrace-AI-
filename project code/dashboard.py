import streamlit as st
from PIL import Image
import tempfile
import os

from main import run_pipeline, process_image_item
from preprocessing import preprocess
from inference import load_processor, load_model


st.set_page_config(page_title="VeriTrace", layout="wide")


@st.cache_resource
def get_model_and_processor():
    #cached so streamlit doesn't reload the model on every rerun/interaction
    processor = load_processor()
    model = load_model()
    return processor, model


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name


def render_image_result(item):
    st.subheader("Answer")
    st.write(item["response"][0] if item["response"] else "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{item['confidence']:.1%}")
    col2.metric("Trust Score", f"{item['trust_result']['trust_score']}/100")
    col3.metric("Trust Level", item["trust_result"]["trust_level"])

    st.subheader("Explainability")
    for key, value in item["explanation"].items():
        with st.expander(key):
            st.text(value)

    heatmap_cols = st.columns(3)
    heatmap_files = [
        ("outputs/heatmaps/image_rollout_overlay.png", "Attention Rollout"),
        ("outputs/heatmaps/image_ig_overlay.png", "Integrated Gradients"),
        ("outputs/heatmaps/image_cross_overlay.png", "Cross Attention"),
    ]
    for col, (path, caption) in zip(heatmap_cols, heatmap_files):
        if os.path.exists(path):
            col.image(path, caption=caption, use_column_width=True)


def main():
    st.title("VeriTrace — Explainable Multimodal AI Auditor")
    st.caption("Upload an image, video, or audio file and ask a question about it.")

    uploaded_file = st.file_uploader(
        "Upload file", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "wav", "mp3", "flac"]
    )
    question = st.text_input("Question", value="What is happening in this file?")

    if uploaded_file and st.button("Run Analysis"):
        with st.spinner("Loading model..."):
            processor, model = get_model_and_processor()

        file_path = save_uploaded_file(uploaded_file)

        with st.spinner("Processing input..."):
            prepared = preprocess(file_path, question, processor)

        modality = prepared["modality"]
        st.info(f"Detected modality: {modality}")

        if modality == "image":
            with st.spinner("Running inference and explainability..."):
                item = process_image_item(processor, model, prepared["data"]["image"], question, source_label="image")
            render_image_result(item)

        else:
            st.warning(
                "Video and audio results are best viewed in the generated report "
                "(outputs/reports/) — running full explainability for every frame "
                "or segment here can take a while."
            )
            with st.spinner(f"Running full {modality} pipeline..."):
                run_pipeline(file_path, question)
            st.success("Done. Check outputs/reports/ for the full report.")


if __name__ == "__main__":
    main()