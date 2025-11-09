-----
# 🔎 Math Proof Verifier

### An ML-powered classifier to determine if a mathematical proof is **Correct** or **Flawed**.

This project uses a **Machine Learning pipeline** to classify text-based mathematical proofs. It is trained on a dataset of \~14,000 examples to distinguish between valid arguments and those containing logical flaws. The model is deployed in an interactive **Streamlit** web application.

-----

## 🚀 Live Demo

Here is the application in action, classifying both a correct and a flawed proof:

| Correct Proof (✅) | Flawed Proof (❌) |
| :---: | :---: |
| \<img src="demo.gif" alt="Demo of a correct proof" width="100%"\> | \<img src="demo1.gif" alt="Demo of a flawed proof" width="100%"\> |

-----

## ✨ Features

  * 🤖 **ML-Powered Classification:** Uses a `LinearSVC` model wrapped in a `TfidfVectorizer` pipeline to understand and classify proof text.
  * 🖥️ **Interactive Web UI:** A simple and clean Streamlit app (`app.py`) for easy interaction.
  * 📊 **Model Confidence:** Displays the model's confidence score for its prediction.
  * 📈 **Rigorous Training:** The model is trained (`train_model.py`) using `GridSearchCV` to find the best-performing hyperparameters.
  * 🎈 **Instant Feedback:** The app celebrates correct proofs (`st.balloons()`) and provides clear warnings for flawed ones.

-----

## 📂 Project Structure

Here is the recommended directory structure for this project.

```
math-proof-verifier/
├── 📦 model/
│   └── proof_verifier_pipeline.joblib   # The saved (trained) model pipeline
│
├── 📊 data/
│   └── math_proof_verifier_dataset_14000_13.csv  # The training dataset
│
├── 📁 assets/
│   ├── demo.gif                         # Demo of a correct proof
│   └── demo1.gif                        # Demo of a flawed proof
│
├── 📜 app.py                             # The Streamlit web application
├── 📜 train_model.py                      # Script to train and save the model
├── 📜 requirements.txt                    # Python dependencies
├── 📜 README.md                           # You are here!
└── 📜 .gitignore                         # Git ignore file
```

-----

## ⚙️ How It Works: The ML Pipeline

The system is broken into two main parts: **Training** and **Inference**.

### 1\. Training (`train_model.py`)

1.  **Load Data:** The `math_proof_verifier_dataset_14000_13.csv` is loaded into a Pandas DataFrame.
2.  **Clean & Split:** The text (`proof_text`) and labels (`is_correct`) are cleaned. The data is then split into 80% training and 20% testing sets.
3.  **Build Pipeline:** A `sklearn.pipeline.Pipeline` is created that first vectorizes the text using `TfidfVectorizer` and then feeds the vectors into a classifier (e.g., `LinearSVC`).
4.  **Hyperparameter Tuning:** `GridSearchCV` is used to test multiple classifiers (Logistic Regression, LinearSVC, Random Forest) and their parameters to find the best-performing model.
5.  **Save Model:** The *entire* best-performing pipeline (vectorizer + classifier) is saved to a single file: `proof_verifier_pipeline.joblib`.

### 2\. Inference (`app.py`)

1.  **Load Model:** The app uses `@st.cache_resource` to load the `proof_verifier_pipeline.joblib` file into memory *only once*.
2.  **User Input:** It provides a `st.text_area` for the user to paste their mathematical proof.
3.  **Predict:** When the "Verify" button is clicked:
      * The model's `.predict()` method is called on the user's text.
      * The model's `.predict_proba()` method is used to get the confidence score.
4.  **Display Results:** The app displays a ✅ or ❌, the verdict, and the model's confidence.

-----

## 🛠️ How to Run This Project Locally

### Prerequisites

  * Python 3.8+
  * `pip` (Python package installer)

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/math-proof-verifier.git
cd math-proof-verifier
```

### 2\. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

You will need to create a `requirements.txt` file. Based on your scripts, it should contain:

```
pandas
numpy
scikit-learn
streamlit
joblib
```

Now, install them:

```bash
pip install -r requirements.txt
```

### 4\. Train the Model (One-time step)

Before you can run the app, you need the `proof_verifier_pipeline.joblib` file. Run the training script:

```bash
python train_model.py
```

*(This will use the `.csv` file to train and save the model in the correct path)*

### 5\. Run the Streamlit App

```bash
streamlit run app.py
```

Your browser should automatically open to the web application\!

-----

## 👥 Contributors

This project was built by the following amazing team.

| Name | Role | GitHub | LinkedIn |
| :--- | :--- | :--- | :--- |
| **[Annadata Aniketh]** | Project Lead & ML Engineer | `https://github.com/Annadata-Aniketh` | `https://www.linkedin.com/in/aniketh-annadata-439667303/` |
| **[Pratheek G N]** | Data Analysis & Model Tuning | `https://github.com/LUCIFER27086` | `https://www.linkedin.com/in/pratheek-g-n-117617358/` |
| **[Akash S]** | Streamlit UI/UX | `TBD` | `TBD` |

-----

## ⚠️ Model Limitations

> **Important:** This is a **text classifier**, not a formal logic verifier or a symbolic reasoner.
>
>   * The model learns from **statistical word patterns** (TF-IDF) in the training data, not from an understanding of mathematical axioms or formal logic.
>   * It **cannot** pinpoint the *specific logical error* in a flawed proof.

>   * It may struggle with novel proof structures or complex mathematical notation not seen in its training data.


