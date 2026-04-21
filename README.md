# Cyberbullying Detection System (ML & Streamlit)

An interactive, web-based Machine Learning application designed to detect and prevent cyberbullying comments in real-time. This project goes beyond static text classification by implementing a **Human-in-the-Loop (Continuous Learning)** architecture, allowing the model to be dynamically corrected and retrained directly through an admin dashboard.

## Key Features
* **Social Media Simulation (Real-time Inference):** Users can type comments, and the system will instantly block or allow the message based on AI predictions.
* **Probability & Confidence Scoring:** Displays the Top 3 violation categories (e.g., *Hate Speech, Abusive*) alongside their calculated confidence scores.
* **Human-in-the-Loop (AI Correction):** A dedicated Admin Dashboard to test sentences, view predictions, and provide feedback (correcting labels if the AI makes a mistake).
* **One-Click Dynamic Retraining:** Admins can trigger model retraining directly from the UI. The system seamlessly merges new feedback data with the main dataset to produce a smarter, updated model.

## Tech Stack & Libraries
* **Language:** Python
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Support Vector Machine)
* **NLP Processing:** TF-IDF Vectorizer (N-grams)
* **Data Handling:** Pandas, Joblib

## Technical Highlights
This project demonstrates proficiency in advanced Machine Learning engineering workflows:

1. **Pipeline Architecture:** Utilizes `sklearn.pipeline.Pipeline` to bundle the `TfidfVectorizer` and the Classifier, preventing data leakage and streamlining the inference process.
2. **Probability Calibration:** Since standard `LinearSVC` does not natively output probabilities, this project implements `CalibratedClassifierCV` (with the sigmoid method) to enable accurate confidence scoring for the "Top 3 Categories" feature.
3. **Advanced State Management:** Leverages Streamlit's `st.session_state` to manage complex admin workflows (Testing -> Correcting -> Saving Feedback) without triggering unwanted app refreshes.
4. **Incremental Learning Simulation:** Uses a file-based temporary queue (`temp_feedback.csv`) to store admin corrections before securely merging them into the main dataset (`DATASET_UTAMA.csv`) during the retraining pipeline.


## Getting Started 

### Prerequisites
Ensure you have Python 3.8+ installed. Using a virtual environment (venv) is highly recommended.

### Installation & Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/Darren0501/AIModel-CyberbUllyingDetector.git
cd cyberbullying-detection

2. Navigate to the project directory:
   ```bash
   cd AIModel-CyberbUllyingDetector

3. Install the required dependencies:
   ```bash
   pip install streamlit pandas joblib scikit-learn

4. Run the application:
   ```bash
   streamlit run MainApp.py
