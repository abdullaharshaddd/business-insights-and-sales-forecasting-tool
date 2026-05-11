import os

files_to_delete = [
    "src/models/lstm_forecaster.py",
    "src/forecasting/prophet_model.py",
    "src/forecasting/feature_engineering.py"
]

def main():
    print("Deleting old forecasting files...")
    for f in files_to_delete:
        path = os.path.join(os.getcwd(), f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted: {path}")
        else:
            print(f"File not found (already deleted): {path}")

if __name__ == "__main__":
    main()
