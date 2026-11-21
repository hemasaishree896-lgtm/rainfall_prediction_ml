import streamlit as st
import xgboost as xgb
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Rainfall Prediction")
st.caption("Enter each weather value once, then predict.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rainfall_xgb.json"


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Keep rainfall_xgb.json in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("The XGBoost model could not be loaded.")
    st.exception(error)
    st.stop()

feature_names = model.get_booster().feature_names

if not feature_names:
    st.error("The model file does not contain feature names.")
    st.stop()


st.subheader("Weather Information")
st.caption(
    "Single values are reused for the model's corresponding time-based fields."
)

with st.form("rainfall_prediction_form"):

    top_left, top_right = st.columns(2)

    with top_left:
        location = st.selectbox(
            "Location",
            ["AliceSprings", "Portland", "Woomera"]
        )

    with top_right:
        rain_today = st.radio(
            "Did it rain today?",
            ["No", "Yes"],
            horizontal=True
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temperature_text = st.text_input(
            "Temperature (°C)",
            value="25"
        )
        rainfall_text = st.text_input(
            "Rainfall (mm)",
            value="0"
        )

    with col2:
        wind_text = st.text_input(
            "Wind speed (km/h)",
            value="15"
        )
        humidity_text = st.text_input(
            "Humidity (%)",
            value="60"
        )

    with col3:
        pressure_text = st.text_input(
            "Pressure (hPa)",
            value="1015"
        )
        sunshine_text = st.text_input(
            "Sunshine (hours)",
            value="8"
        )

    with col4:
        cloud_text = st.text_input(
            "Cloud cover (0–8)",
            value="4"
        )
        evaporation_text = st.text_input(
            "Evaporation (mm)",
            value="5"
        )

    submitted = st.form_submit_button(
        "Predict Rainfall",
        use_container_width=True
    )


def read_number(label, value, minimum=None, maximum=None):
    try:
        number = float(value.strip())
    except (ValueError, AttributeError):
        raise ValueError(
            f"Please enter a valid number for {label}."
        )

    if not pd.notna(number):
        raise ValueError(
            f"{label} must be a valid number."
        )

    if minimum is not None and number < minimum:
        raise ValueError(
            f"{label} must be at least {minimum}."
        )

    if maximum is not None and number > maximum:
        raise ValueError(
            f"{label} must be at most {maximum}."
        )

    return number


if submitted:
    try:
        temperature = read_number(
            "Temperature",
            temperature_text
        )

        rainfall = read_number(
            "Rainfall",
            rainfall_text,
            minimum=0
        )

        wind = read_number(
            "Wind speed",
            wind_text,
            minimum=0
        )

        humidity = read_number(
            "Humidity",
            humidity_text,
            minimum=0,
            maximum=100
        )

        pressure = read_number(
            "Pressure",
            pressure_text,
            minimum=0
        )

        sunshine = read_number(
            "Sunshine",
            sunshine_text,
            minimum=0,
            maximum=24
        )

        cloud = read_number(
            "Cloud cover",
            cloud_text,
            minimum=0,
            maximum=8
        )

        evaporation = read_number(
            "Evaporation",
            evaporation_text,
            minimum=0
        )

        row = {
            "MinTemp": temperature,
            "MaxTemp": temperature,
            "Rainfall": rainfall,
            "Evaporation": evaporation,
            "Sunshine": sunshine,
            "WindGustSpeed": wind,
            "WindSpeed9am": wind,
            "WindSpeed3pm": wind,
            "Humidity9am": humidity,
            "Humidity3pm": humidity,
            "Pressure9am": pressure,
            "Pressure3pm": pressure,
            "Cloud9am": cloud,
            "Cloud3pm": cloud,
            "Temp3pm": temperature,
            "Location_AliceSprings": int(
                location == "AliceSprings"
            ),
            "Location_Portland": int(
                location == "Portland"
            ),
            "Location_Woomera": int(
                location == "Woomera"
            ),
            "RainToday_No": int(
                rain_today == "No"
            ),
            "RainToday_Yes": int(
                rain_today == "Yes"
            )
        }

        input_df = pd.DataFrame(
            [row],
            columns=feature_names
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🌧️ Rainfall: YES")
        else:
            st.success("☀️ Rainfall: NO")

        if 1 in classes:
            rain_probability = (
                float(probabilities[classes.index(1)]) * 100
            )

            st.metric(
                "Rainfall probability",
                f"{rain_probability:.2f}%"
            )

            st.progress(
                min(
                    max(rain_probability / 100, 0.0),
                    1.0
                )
            )

    except ValueError as error:
        st.warning(str(error))

    except Exception as error:
        st.error(
            "Prediction failed. Check that the model "
            "and feature names match."
        )
        st.exception(error)


st.markdown("---")

with st.expander("View the model's most important features"):
    try:
        importance = model.get_booster().get_score(
            importance_type="gain"
        )

        if importance:
            importance_df = (
                pd.DataFrame({
                    "Feature": list(importance.keys()),
                    "Importance (gain)": list(importance.values())
                })
                .sort_values(
                    "Importance (gain)",
                    ascending=False
                )
                .head(10)
                .set_index("Feature")
            )

            st.bar_chart(importance_df)

        else:
            st.write(
                "Feature importance information is not available."
            )

    except Exception:
        st.write(
            "Feature importance information could not be displayed."
        )


st.caption(
    "This app uses your trained XGBoost model. The prediction "
    "is an estimate and depends on the model's training data "
    "and entered weather values."
)