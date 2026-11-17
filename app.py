
import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_large, resnet18
from torchvision import transforms

# 페이지 설정
st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 디자인
st.markdown("""
<style>
/* 앱 전체 회색 배경 */
html,
body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background-color: #303030 !important;
    color: #F5F5F5 !important;
}

[data-testid="stMain"] > div,
.block-container {
    background-color: #303030 !important;
    color: #F5F5F5 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* 사이드바 */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #252525 !important;
    color: #F5F5F5 !important;
}

/* 기본 글자 */
.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #F5F5F5;
}

/* 제목 */
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #FFFFFF !important;
    margin-bottom: 5px;
}

.sub-title {
    color: #D4D4D4 !important;
    font-size: 15px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #F5F5F5 !important;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* 안내 카드 */
.info-card {
    background-color: #424242;
    border: 1px solid #666666;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 15px;
}

.info-card h3,
.info-card p {
    color: #F5F5F5 !important;
}

/* 업로더 바탕은 투명하게 */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* 업로드 안내 문구와 찾아보기 버튼 */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] button {
    color: #FFFFFF !important;
}

/* 업로드된 파일 칩 배경 */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] > div {
    background-color: #D1D1D1 !important;
    border-radius: 6px !important;
}

/* 파일명과 파일 용량은 검정색 */
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileSize"] {
    color: #111111 !important;
}

/* 파일 칩 안의 아이콘도 어둡게 */
[data-testid="stFileUploaderFile"] svg {
    color: #333333 !important;
    fill: #333333 !important;
}

/* 업로드 영역의 + 아이콘 숨기기 */
[data-testid="stFileUploader"] button[aria-label*="Add"],
[data-testid="stFileUploader"] button[title*="Add"] {
    display: none !important;
}

/* 업로드 영역의 장식용 아이콘 숨기기 */
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    display: none !important;
}

/* 웹캠 버튼 */
.stButton button,
[data-testid="stCameraInput"] button {
    background-color: #555555 !important;
    color: #FFFFFF !important;
    border: 1px solid #777777 !important;
}

/* 1위 결과 카드 */
.top-result {
    background-color: #FFFFFF;
    color: #111111 !important;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 18px;
    font-weight: 700;
}

.top-result * {
    color: #111111 !important;
}

/* 오른쪽 위 메뉴와 푸터 숨기기 */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# 모델 선택 및 불러오기
MODEL_DIR = Path("CNN_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILES = sorted([p for p in MODEL_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pt", ".pth"}], key=lambda p: (p.name != "YOLO.pt", p.name.lower()))
if not MODEL_FILES:
    st.error("CNN_models 폴더에 .pt 또는 .pth 모델 파일이 없습니다.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "사용할 모델", MODEL_FILES,
    index=next((i for i, p in enumerate(MODEL_FILES) if p.name == "YOLO.pt"), 0),
    format_func=lambda path: path.name,
)

@st.cache_resource
def load_model(model_path):
    model_path = Path(model_path)
    if model_path.suffix.lower() == ".pt":
        return {"kind": "yolo", "model": YOLO(str(model_path))}

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    class_names = checkpoint.get("class_names") or checkpoint.get("classes")
    if not class_names:
        raise ValueError("ResNet18 체크포인트에 class_names 정보가 없습니다. 학습 노트북에서 모델을 다시 저장하세요.")
    if any(key.startswith("classifier.") for key in state):
        model = mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
        kind = "classifier"
    else:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        kind = "classifier"

    model.load_state_dict(state)
    model.eval()
    return {"kind": kind, "model": model, "class_names": class_names}

try:
    model_bundle = load_model(str(selected_model))
except Exception as e:
    st.error(f"모델을 불러오지 못했습니다: {e}")
    st.stop()

# ResNet18 학습 시 사용한 입력 전처리
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 두 모델 모두 같은 {name, confidence} 형식의 Top-5 결과 반환
def predict_image(image_rgb):
    if model_bundle["kind"] == "classifier":
        image_tensor = resnet_transform(Image.fromarray(image_rgb)).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model_bundle["model"](image_tensor), dim=1)[0]
        scores, ids = torch.topk(probabilities, k=min(5, len(model_bundle["class_names"])))
        return [
            {"name": model_bundle["class_names"][int(class_id)], "confidence": float(score)}
            for score, class_id in zip(scores, ids)
        ]

    result = model_bundle["model"](image_rgb, verbose=False)[0]
    predictions = []
    if result.probs is not None:
        for class_id, confidence in zip(result.probs.top5, result.probs.top5conf):
            predictions.append({"name": result.names[int(class_id)], "confidence": float(confidence)})
    elif result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            predictions.append({"name": result.names[class_id], "confidence": float(box.conf[0])})
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        predictions = predictions[:5]
    return predictions


# Top-5 그래프와 1위 결과 표시
def show_predictions(predictions):
    if not predictions:
        st.warning("예측 결과가 없습니다.")
        return

    st.markdown("### 📊 Top-5 예측 결과")

    chart_data = pd.DataFrame([
        {
            "피부질환": item["name"],
            "확률(%)": item["confidence"] * 100,
        }
        for item in predictions
    ])

    # 파랑·청회색 팔레트
    palette = [
        "#5B8CCB",
        "#37699F",
        "#536B86",
        "#7E8FA3",
        "#A9C5E3",
    ]

    color_scale = alt.Scale(
        domain=chart_data["피부질환"].tolist(),
        range=palette[:len(chart_data)],
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "확률(%):Q",
                title="확률(%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "피부질환:N",
                title=None,
                sort="-x",
            ),
            color=alt.Color(
                "피부질환:N",
                scale=color_scale,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("피부질환:N", title="피부질환"),
                alt.Tooltip(
                    "확률(%):Q",
                    format=".2f",
                    title="확률(%)",
                ),
            ],
        )
        .properties(height=max(160, 42 * len(chart_data)))
    )

    st.altair_chart(chart, use_container_width=True)

    top_prediction = predictions[0]
    st.markdown(
        f"""
        <div class="top-result">
            1위 예측: {top_prediction['name']}
            ({top_prediction['confidence'] * 100:.2f}%)
        </div>
        """,
        unsafe_allow_html=True,
    )


# 사이드바
with st.sidebar:
    st.markdown("## 🔬 SkinCare AI")
    st.caption("AI Skin Disease Analysis")
    st.divider()
    st.markdown("### 📌 메뉴")

    mode = st.radio(
        "분석 방식",
        ["이미지 분석", "웹캠 분석"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 🤖 AI Model")
    st.markdown(f"**선택 모델:** `{selected_model.name}`\n\n모델을 바꾸면 이미지/웹캠 분석에 즉시 적용됩니다.")

    st.divider()
    st.caption("SkinCare AI v1.0")


# 메인 화면
st.markdown(
    '<div class="main-title">🔬 AI 피부질환 분석 서비스</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'YOLO 또는 ResNet18 기반 인공지능 모델을 활용한 피부 이미지 분석 서비스입니다.'
    '</div>',
    unsafe_allow_html=True,
)


# 이미지 분석
if mode == "이미지 분석":
    st.markdown(
        '<div class="section-title">📤 피부 이미지 업로드</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "분석할 피부 이미지를 선택하세요.",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        st.markdown("### 🖼️ 업로드 이미지")
        st.image(image_rgb, use_container_width=True)

        with st.spinner("AI 모델이 피부 이미지를 분석하고 있습니다..."):
            predictions = predict_image(image_rgb)

        st.divider()
        show_predictions(predictions)

    else:
        st.markdown("""
        <div class="info-card">
            <h3>👋 SkinCare AI에 오신 것을 환영합니다.</h3>
            <p>
                피부 이미지를 업로드하면 YOLOv11 모델이 분석하고
                Top-5 예측 결과를 그래프로 보여줍니다.
            </p>
            <p>📌 지원 이미지: JPG / JPEG / PNG</p>
        </div>
        """, unsafe_allow_html=True)


# 웹캠 분석
elif mode == "웹캠 분석":
    st.markdown("### 📷 웹캠 피부 이미지 분석")

    camera_image = st.camera_input("피부 이미지를 촬영하세요.")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_rgb = np.array(image)

        with st.spinner("AI 모델이 피부 이미지를 분석하고 있습니다..."):
            predictions = predict_image(image_rgb)

        st.markdown("### 🖼️ 촬영 이미지")
        st.image(image_rgb, use_container_width=True)

        show_predictions(predictions)


# 하단 안내
st.divider()

st.markdown("""
<div style="
    background-color: #303030;
    text-align: center;
    color: #D4D4D4;
    font-size: 12px;
">
    SkinCare AI | Powered by YOLOv11<br>
    본 서비스는 AI 기반 참고용 분석 결과를 제공하며,
    의료진의 진단을 대체하지 않습니다.
</div>
""", unsafe_allow_html=True)
