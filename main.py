import sys
import os
import subprocess


def main():
    print("Student Intelligence Platform")
    print("==============================")

    # step 1 — only seed the database if it doesn't exist yet
    # this avoids re-downloading and re-inserting data every time
    if not os.path.exists("data/sip.db"):
        print("\n[1/3] Seeding database from Kaggle dataset...")
        from data.seed import seed
        seed()
    else:
        print("\n[1/3] Database already exists, skipping seed.")

    # step 2 — only train the models if no saved models exist
    # training is skipped after the first run since models are saved to disk
    if not os.path.exists("models/grade_predictor.pkl"):
        print("[2/3] Training ML models...")
        from src.models import train_and_evaluate
        train_and_evaluate()
    else:
        print("[2/3] Trained models found, skipping training.")

    # step 3 — launch the streamlit dashboard in the browser
    print("[3/3] Launching dashboard...")
    print("\nOpen your browser at: http://localhost:8501\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])


if __name__ == "__main__":
    main()
