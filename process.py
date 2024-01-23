from data.data_vis import clean_data

PATH = "data/train.xlsx"

def main(path: str) -> None:
    # Load the dataset
    df = clean_data(path)

    # Compute moving averages
    df["Moving Average"] = df["Demand"].rolling(window=10).mean()

if __name__ == "__main__":
    main(PATH)