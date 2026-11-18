"""Project-wide constants (mirrors the notebook setup)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
RS = 42
TEST_SIZE = 0.2

# Palette (dark UI)
C_DT = "#38BDF8"     # decision tree  - sky
C_RF = "#34D399"     # random forest  - emerald
C_ACC = "#A78BFA"    # accent         - violet
C_RED = "#F87171"    # warning / train
C_AMB = "#FBBF24"    # highlight

FEATURE_INFO = {
    "age": ("Age", "Age in years"),
    "sex": ("Sex", "1 = male, 0 = female"),
    "cp": ("Chest pain type", "Coded 0-3"),
    "trestbps": ("Resting BP", "Resting blood pressure (mm Hg)"),
    "chol": ("Cholesterol", "Serum cholesterol (mg/dl)"),
    "fbs": ("Fasting blood sugar", "> 120 mg/dl (1 = true)"),
    "restecg": ("Resting ECG", "Coded 0-2"),
    "thalach": ("Max heart rate", "Maximum heart rate achieved"),
    "exang": ("Exercise angina", "1 = yes"),
    "oldpeak": ("ST depression", "Induced by exercise vs rest"),
    "slope": ("ST slope", "Slope of the peak exercise ST segment"),
    "ca": ("Major vessels", "Number coloured by fluoroscopy (0-4)"),
    "thal": ("Thalassemia", "Coded 0-3"),
}
CLASS_NAMES = ["target 0", "target 1"]

# Tuned hyper-parameters found by grid search in the notebook
DT_TUNED = dict(criterion="entropy", max_depth=3, min_samples_leaf=10, ccp_alpha=0.02)
RF_TUNED = dict(n_estimators=300, max_depth=8, max_features="sqrt", min_samples_leaf=1)
