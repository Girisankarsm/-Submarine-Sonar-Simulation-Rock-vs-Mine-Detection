# p.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
import time
import os
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.align import Align

console = Console()

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Load Dataset ----------
def load_sonar_data(path="sonar.alldata.txt") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found.")

    df = pd.read_csv(path, header=None, sep=",", skip_blank_lines=True)
    if df.shape[0] == 0 or df.shape[1] == 1:
        df = pd.read_csv(path, header=None, sep="\s+", skip_blank_lines=True, engine="python")
    if df.shape[1] != 61:
        raise ValueError(f"Expected 61 columns, got {df.shape[1]}")

    cols = [f"V{i}" for i in range(1, 61)] + ["Label"]
    df.columns = cols

    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    if df.iloc[:, :-1].isnull().any().any():
        raise ValueError("Dataset contains non-numeric values in feature columns.")

    df["Label"] = df["Label"].map({"R": 0, "M": 1})
    if df["Label"].isnull().any():
        raise ValueError("Dataset contains invalid labels")

    logger.info(f"Loaded dataset with shape {df.shape}")
    return df

# ---------- Train Model ----------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    logger.info("✅ Model training completed.")
    return model

# ---------- Evaluate Model ----------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"📊 Test Accuracy: {acc*100:.2f}%")
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

# ---------- Interactive Ocean Simulation ----------
def interactive_ocean(model, width=40, height=10, num_scans=80, delay=0.2):
    # Initialize moving objects
    num_objects = width // 2
    objects = []
    for _ in range(num_objects):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        obj_type = np.random.choice([0, 1])  # 0=Rock,1=Mine
        dx = np.random.choice([-1, 1])       # horizontal movement
        dy = np.random.choice([-1, 0, 1])    # vertical movement
        objects.append({"x": x, "y": y, "type": obj_type, "dx": dx, "dy": dy})

    submarine = "🚢"
    sub_y = height // 2

    console.print("\n🚀 Starting Interactive Ocean Simulation...\n", style="bold cyan")
    with Live(refresh_per_second=10) as live:
        for scan in range(num_scans):
            grid = [["~" for _ in range(width)] for _ in range(height)]
            sub_x = scan % width
            grid[sub_y][sub_x] = submarine

            # Move objects
            for obj in objects:
                obj["x"] = (obj["x"] + obj["dx"]) % width
                obj["y"] = max(0, min(height - 1, obj["y"] + obj["dy"]))

                # Sonar detection radius
                if abs(obj["x"] - sub_x) <= 3 and abs(obj["y"] - sub_y) <= 3:
                    symbol = "🟢" if obj["type"] == 0 else "💣"
                    grid[obj["y"]][obj["x"]] = symbol

            # Build display
            display = "\n".join("".join(row) for row in grid)
            panel = Panel(Align.center(display), border_style="bright_blue")
            live.update(panel)
            time.sleep(delay)

# ---------- Main ----------
def main():
    df = load_sonar_data("sonar.alldata.txt")
    X = df.drop("Label", axis=1)
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sonar_rf.joblib")
    logger.info("💾 Model saved to models/sonar_rf.joblib")

    interactive_ocean(model, width=40, height=10, num_scans=80, delay=0.2)

# ---------- Entry ----------
if __name__ == "__main__":
    main()
