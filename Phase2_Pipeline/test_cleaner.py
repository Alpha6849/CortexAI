from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector
from pipeline.cleaner import DataCleaner

csv_path = r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv"


def print_section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)


try:
    loader = DataLoader(csv_path)
    df, meta = loader.load()

    # Detect schema
    schema = SchemaDetector(df).detect()

    print_section("Schema Detected")
    print(schema)

    # Initialize Cleaner
    cleaner = DataCleaner(df, schema)

    print_section("Cleaner Initialized")
    print("Cleaner DataFrame Shape:", cleaner.df.shape)
    print("Cleaner Report:", cleaner.report)
    
    print_section("Running Cleaner")
    
    cleaned_df = cleaner.clean()
    print(cleaned_df.head())
    print("\nCleaning Report:", cleaner.report)


except Exception as e:
    print("\n Error:")
    print(e)
