from pipeline.loader import DataLoader

# trying on my sample iris csv 
csv_path = r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv"

try:
    loader = DataLoader(csv_path)

    df, meta = loader.load()   # <-- unpack the tuple

    print("\n CSV Loaded Successfully!")
    print("Shape:", df.shape)
    print("\n Metadata:")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nFirst 5 rows:")
    print(df.head())

except Exception as e:
    print("\n Error while loading CSV:")
    print(e)
