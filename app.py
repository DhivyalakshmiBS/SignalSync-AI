import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SignalSync AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "signalsense_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    st.error(
        "Model file not found. Please keep signalsense_model.pkl "
        "in the project folder."
    )
    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(67, 97, 238, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 200, 255, 0.07),
            transparent 30%
        ),
        #080b12;
    color: #f5f7fb;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0c1019;
    border-right: 1px solid #202735;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #334155;
    background: linear-gradient(135deg, #3155d9, #5936c9);
    color: white;
    font-weight: 700;
    padding: 0.7rem 1rem;
    transition: 0.2s;
}

.stButton > button:hover {
    border-color: #6d8cff;
    transform: translateY(-1px);
}


/* Upload box */

[data-testid="stFileUploader"] {
    background: #101621;
    border: 1px dashed #42516b;
    border-radius: 14px;
    padding: 10px;
}


/* Metric cards */

[data-testid="stMetric"] {
    background: #101621;
    border: 1px solid #202a3a;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #8d9bb2 !important;
}

[data-testid="stMetricValue"] {
    color: #f5f7fb !important;
}


/* Divider */

hr {
    border-color: #202735 !important;
}


/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #202a3a;
    border-radius: 12px;
}


/* Selectbox */

div[data-baseweb="select"] > div {
    background-color: #101621;
    border-color: #303b50;
    border-radius: 10px;
}


/* Info cards */

.info-card {
    background: linear-gradient(135deg, #101722, #0d131e);
    border: 1px solid #202b3c;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}


/* Hero */

.hero {
    padding: 10px 0 25px 0;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 5px;
}

.hero-title span {
    color: #6f8cff;
}

.hero-subtitle {
    color: #91a0b8;
    font-size: 16px;
    line-height: 1.6;
}


/* Section title */

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin: 20px 0 12px 0;
}


/* Small label */

.small-label {
    color: #8290a7;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}


/* Status */

.status-normal {
    color: #48d597;
    font-size: 25px;
    font-weight: 800;
}

.status-warning {
    color: #f5bd4f;
    font-size: 25px;
    font-weight: 800;
}

.status-critical {
    color: #ff6577;
    font-size: 25px;
    font-weight: 800;
}


/* Badges */

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.badge-green {
    background: rgba(72,213,151,0.12);
    color: #48d597;
    border: 1px solid rgba(72,213,151,0.25);
}

.badge-yellow {
    background: rgba(245,189,79,0.12);
    color: #f5bd4f;
    border: 1px solid rgba(245,189,79,0.25);
}

.badge-red {
    background: rgba(255,101,119,0.12);
    color: #ff6577;
    border: 1px solid rgba(255,101,119,0.25);
}


/* Footer */

.footer-text {
    text-align: center;
    color: #59677e;
    font-size: 12px;
    padding: 25px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(signal):

    signal = np.asarray(signal, dtype=float)

    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.ptp(signal),
        np.sqrt(np.mean(signal ** 2)),
        np.mean(np.abs(np.diff(signal))),
        np.median(signal)
    ]

    fft_values = np.abs(np.fft.rfft(signal))

    features.extend([
        np.mean(fft_values),
        np.std(fft_values),
        np.max(fft_values),
        np.argmax(fft_values)
    ])

    return np.array(features)


# =========================================================
# READ UPLOADED CSV
# =========================================================

def read_waveform(uploaded_file):

    data = pd.read_csv(
        uploaded_file,
        header=None
    )

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.dropna(
        axis=0,
        how="all"
    )

    data = data.dropna(
        axis=1,
        how="all"
    )

    if len(data) > 1:

        numeric_values = data.values.flatten()

        numeric_values = numeric_values[
            ~np.isnan(numeric_values)
        ]

        signal = numeric_values[:100]

    else:

        signal = data.iloc[0].dropna().values

    return np.asarray(
        signal,
        dtype=float
    )


# =========================================================
# SEVERITY
# =========================================================

def get_severity(prediction):

    if prediction == "Pure_Sinusoidal":
        return "NORMAL", "green"

    critical = [
        "Interruption",
        "Transient",
        "Notch"
    ]

    high = [
        "Sag",
        "Swell",
        "Sag_with_Harmonics",
        "Swell_with_Harmonics",
        "Flicker_with_Sag",
        "Flicker_with_Swell"
    ]

    if prediction in critical:
        return "CRITICAL", "red"

    if prediction in high:
        return "HIGH", "yellow"

    return "MEDIUM", "yellow"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:28px;font-weight:800;">
        ⚡ SignalSync AI
        </div>

        <div style="color:#7f8da5;font-size:13px;margin-top:3px;">
        Intelligent Power Quality Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📂 Analyze Signal")

    uploaded_file = st.file_uploader(
        "Upload waveform CSV",
        type=["csv"],
        help="Upload a CSV containing an electrical waveform."
    )

    st.divider()

    st.markdown("### 🔍 How It Works")

    st.markdown(
        """
        <div class="info-card">

        <div style="font-size:15px;font-weight:700;">
        01&nbsp;&nbsp; Upload
        </div>

        <div style="color:#8d9bb2;font-size:13px;
        line-height:1.6;margin-top:5px;">
        Upload an electrical waveform CSV for analysis.
        </div>


        <div style="font-size:15px;font-weight:700;
        margin-top:18px;">
        02&nbsp;&nbsp; Analyze
        </div>

        <div style="color:#8d9bb2;font-size:13px;
        line-height:1.6;margin-top:5px;">
        The system extracts signal characteristics
        from the waveform.
        </div>


        <div style="font-size:15px;font-weight:700;
        margin-top:18px;">
        03&nbsp;&nbsp; Detect
        </div>

        <div style="color:#8d9bb2;font-size:13px;
        line-height:1.6;margin-top:5px;">
        AI identifies the most likely power-quality
        disturbance.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Analysis Output")

    st.markdown(
        """
        <div class="info-card">

        <div style="color:#b6c1d2;
        font-size:13px;line-height:1.8;">

        • Detected disturbance<br>
        • Signal severity<br>
        • AI confidence<br>
        • Prediction distribution<br>
        • Signal characteristics

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

    <div class="hero-title">
    ⚡ Signal<span>Sync AI</span>
    </div>

    <div class="hero-subtitle">
    AI-powered electrical waveform intelligence for
    automated power quality disturbance detection.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)




# =========================================================
# UPLOAD / ANALYSIS AREA
# =========================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-card"
        style="text-align:center;padding:55px 30px;">

        <div style="font-size:45px;">
        📡
        </div>

        <div style="font-size:24px;
        font-weight:700;margin-top:10px;">
        Upload an Electrical Waveform
        </div>

        <div style="color:#8290a7;margin-top:8px;">
        Upload a CSV signal to begin
        AI-powered power quality analysis.
        </div>

        <div style="color:#56647b;
        font-size:13px;margin-top:18px;">
        Supported format: CSV • Waveform length: 100 samples
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


else:

    try:

        # =================================================
        # READ SIGNAL
        # =================================================

        signal = read_waveform(
            uploaded_file
        )

        if len(signal) < 10:

            st.error(
                "The uploaded CSV does not contain "
                "enough waveform samples."
            )

            st.stop()

        # Training uses first 100 samples

        signal = signal[:100]


        # =================================================
        # UPLOADED SIGNAL CARD
        # =================================================

        st.markdown(
            f"""
            <div class="info-card">

            <div class="small-label">
            Uploaded Signal
            </div>

            <div style="font-size:20px;
            font-weight:700;margin-top:5px;">

            📄 {uploaded_file.name}

            </div>

            <div style="color:#718099;
            font-size:13px;margin-top:5px;">

            {len(signal)} samples detected

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # ANALYZE BUTTON
        # =================================================

        analyze = st.button(
            "⚡ ANALYZE WAVEFORM",
            use_container_width=True
        )


        # =================================================
        # WAVEFORM
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Waveform Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        chart_col, stats_col = st.columns(
            [2.5, 1]
        )


        with chart_col:

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            fig.patch.set_facecolor(
                "#101621"
            )

            ax.set_facecolor(
                "#101621"
            )

            ax.plot(
                range(len(signal)),
                signal,
                linewidth=2
            )

            ax.set_title(
                "Uploaded Electrical Signal",
                color="white",
                fontsize=13
            )

            ax.set_xlabel(
                "Sample",
                color="#8795aa"
            )

            ax.set_ylabel(
                "Amplitude",
                color="#8795aa"
            )

            ax.tick_params(
                colors="#8795aa"
            )

            for spine in ax.spines.values():

                spine.set_color(
                    "#273244"
                )

            ax.grid(
                alpha=0.15
            )

            st.pyplot(
                fig,
                use_container_width=True
            )


        with stats_col:

            st.markdown(
                """
                <div class="info-card">

                <div class="small-label">
                Signal Statistics
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Maximum",
                f"{np.max(signal):.4f}"
            )

            st.metric(
                "Minimum",
                f"{np.min(signal):.4f}"
            )

            st.metric(
                "Mean",
                f"{np.mean(signal):.4f}"
            )

            st.metric(
                "RMS",
                f"{np.sqrt(np.mean(signal**2)):.4f}"
            )


        # =================================================
        # AI RESULT
        # =================================================

        if analyze:

            features = extract_features(
                signal
            )

            prediction = model.predict(
                features.reshape(1, -1)
            )[0]

            probabilities = model.predict_proba(
                features.reshape(1, -1)
            )[0]

            confidence = (
                np.max(probabilities) * 100
            )

            severity, severity_color = get_severity(
                prediction
            )


            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🧠 AI Detection Result'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # RESULT STYLE
            # =================================================

            if severity_color == "green":

                status_class = "status-normal"
                badge_class = "badge-green"

            elif severity_color == "red":

                status_class = "status-critical"
                badge_class = "badge-red"

            else:

                status_class = "status-warning"
                badge_class = "badge-yellow"


            # =================================================
            # RESULT CARDS
            # =================================================

            r1, r2, r3 = st.columns(
                [1.5, 1, 1]
            )


            with r1:

                st.markdown(
                    f"""
                    <div class="info-card">

                    <div class="small-label">
                    Detected Disturbance
                    </div>

                    <div class="{status_class}"
                    style="margin-top:8px;">

                    {prediction.replace("_", " ")}

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r2:

                st.markdown(
                    f"""
                    <div class="info-card">

                    <div class="small-label">
                    Severity
                    </div>

                    <div style="margin-top:12px;">

                    <span class="badge {badge_class}">
                    {severity}
                    </span>

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r3:

                st.markdown(
                    f"""
                    <div class="info-card">

                    <div class="small-label">
                    AI Confidence
                    </div>

                    <div style="font-size:25px;
                    font-weight:800;margin-top:8px;">

                    {confidence:.2f}%

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🎯 Prediction Confidence'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(confidence / 100, 1.0)
            )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🔎 Top Predictions'
                '</div>',
                unsafe_allow_html=True
            )

            class_names = model.classes_

            probability_df = pd.DataFrame({

                "Disturbance": [
                    name.replace("_", " ")
                    for name in class_names
                ],

                "Confidence (%)":
                    probabilities * 100

            })


            probability_df = (
                probability_df
                .sort_values(
                    "Confidence (%)",
                    ascending=False
                )
                .head(5)
            )


            probability_df[
                "Confidence (%)"
            ] = probability_df[
                "Confidence (%)"
            ].round(2)


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # EXTRACTED FEATURES
            # =================================================

            with st.expander(
                "🔬 View extracted signal features"
            ):

                feature_names = [

                    "Mean",
                    "Standard Deviation",
                    "Maximum",
                    "Minimum",
                    "Peak-to-Peak",
                    "RMS",
                    "Average Signal Change",
                    "Median",
                    "FFT Mean",
                    "FFT Standard Deviation",
                    "FFT Maximum",
                    "Dominant Frequency Index"

                ]


                feature_table = pd.DataFrame({

                    "Feature":
                        feature_names,

                    "Value":
                        features

                })


                feature_table["Value"] = (
                    feature_table["Value"]
                    .round(5)
                )


                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # INTERPRETATION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '💡 AI Interpretation'
                '</div>',
                unsafe_allow_html=True
            )


            interpretations = {

                "Pure_Sinusoidal":
                    "The waveform closely resembles a clean "
                    "sinusoidal signal with no major detected "
                    "disturbance.",

                "Sag":
                    "The waveform shows a reduction in voltage "
                    "magnitude, indicating a possible voltage "
                    "sag event.",

                "Swell":
                    "The waveform shows an increase in voltage "
                    "magnitude, indicating a possible voltage "
                    "swell event.",

                "Interruption":
                    "The signal contains a significant "
                    "interruption pattern that may indicate "
                    "temporary loss of supply.",

                "Transient":
                    "A short-duration high-frequency "
                    "disturbance pattern has been detected.",

                "Harmonics":
                    "The waveform contains harmonic components "
                    "that deviate from the fundamental signal.",

                "Flicker":
                    "The signal exhibits low-frequency voltage "
                    "variation characteristics associated "
                    "with flicker.",

                "Notch":
                    "The waveform contains a repetitive "
                    "notch-like disturbance.",

                "Oscillatory_Transient":
                    "The signal contains a short-duration "
                    "oscillatory disturbance."
            }


            explanation = interpretations.get(

                prediction,

                "The AI model detected a power-quality "
                "disturbance pattern based on the learned "
                "waveform characteristics."

            )


            st.markdown(
                f"""
                <div class="info-card">

                <div style="font-size:15px;
                line-height:1.7;
                color:#b6c1d2;">

                {explanation}

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error(
            f"Error while analyzing waveform: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-text">

    SignalSync AI • Machine Learning for
    Intelligent Power Quality Monitoring

    <br>

    AIML × Electrical & Communication Engineering

    </div>
    """,
    unsafe_allow_html=True
)