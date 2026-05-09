import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==============================
# Load the Trained Model
# ==============================
class HRSModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(HRSModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model
model = HRSModel(input_dim=13)  # update input_dim if your dataset differs
model.load_state_dict(torch.load("federated_hrs_model.pth", map_location=torch.device('cpu')))
model.eval()

# ==============================
# Streamlit Interface
# ==============================
st.title("🏥 Federated Healthcare Recommendation System")
st.markdown("### Predict Diabetes Risk using Privacy-Preserving AI")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=120, value=32)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.3)
hbA1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)
smoking = st.selectbox("Smoking History", ["never", "former", "current"])

# Encode categorical inputs
gender_val = 1 if gender == "Male" else 0
smoking_map = {"never": 0, "former": 1, "current": 2}
smoking_val = smoking_map[smoking]

# Create feature vector (adjust based on your dataset’s columns)
input_data = np.array([[gender_val, age, bmi, hbA1c, glucose, smoking_val,
                        0, 0, 1, 0, 0, 0, 0]])  # placeholders for other features
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# ==============================
# Prediction and Recommendations
# ==============================
if st.button("Get Recommendation"):
    with torch.no_grad():
        output = F.softmax(model(input_tensor), dim=1)
        pred = torch.argmax(output, axis=1).item()
        prob = output[0][pred].item() * 100

    label = "🩸 Diabetic" if pred == 1 else "💚 Non-Diabetic"
    st.subheader(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {prob:.2f}%")

    # ------------------------------
    # Personalized Data-Driven Recommendations
    # ------------------------------
    st.markdown("### 🧠 Personalized Health Recommendations")

    # General health tips based on inputs
    recommendations = []

    if bmi > 30:
        recommendations.append("Your BMI indicates obesity. Consider a balanced diet and regular exercise to reduce diabetes risk.")
    elif bmi < 18.5:
        recommendations.append("Your BMI is below normal. A nutrition-rich diet is recommended to maintain a healthy weight.")
    else:
        recommendations.append("Your BMI is within a healthy range. Continue maintaining a balanced diet and exercise routine.")

    if hbA1c >= 6.5:
        recommendations.append("Your HbA1c level is high. Regular glucose monitoring and consultation with a doctor are advised.")
    elif 5.7 <= hbA1c < 6.5:
        recommendations.append("You are in the prediabetic range. Focus on diet and exercise to prevent diabetes onset.")
    else:
        recommendations.append("Your HbA1c level is within the normal range. Keep up your current lifestyle habits.")

    if glucose >= 140:
        recommendations.append("Elevated blood glucose detected. Monitor sugar intake and seek medical guidance.")
    elif glucose < 70:
        recommendations.append("Your glucose level is low. Ensure balanced meals to maintain stable glucose levels.")
    else:
        recommendations.append("Your blood glucose level is normal. Maintain this through regular health checks.")

    if smoking == "current":
        recommendations.append("Smoking increases diabetes complications. Reducing or quitting smoking can improve long-term health.")
    elif smoking == "former":
        recommendations.append("Good progress on quitting smoking! Continue to maintain a smoke-free lifestyle.")

    if age > 50:
        recommendations.append("Age is a diabetes risk factor. Regular check-ups and physical activity are highly recommended.")
    elif age < 25:
        recommendations.append("While young, maintaining a healthy diet and active lifestyle reduces long-term diabetes risk.")

    # Display all recommendations
    for rec in recommendations:
        st.info(rec)

    # Additional note for diabetic prediction
    if pred == 1:
        st.warning("⚠️ Please consult a certified healthcare provider for personalized medical advice and follow-up diagnostics.")

# Footer
st.markdown("---")
st.markdown("This prototype demonstrates a privacy-aware Federated Healthcare Recommendation System using Federated Learning and Explainable AI.")
