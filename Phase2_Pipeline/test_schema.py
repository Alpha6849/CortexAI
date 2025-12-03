from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector

# ----------------------------------------
# CHANGE THIS PATH TO ANY CSV YOU WANT
# ----------------------------------------
csv_path = r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv"


def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


try:
    # Load CSV using our DataLoader
    loader = DataLoader(csv_path)
    df, meta = loader.load()

    print_section("CSV Loaded Successfully")
    print(df.head())
    print("\nMetadata:", meta)

    # Initialize schema detector
    schema = SchemaDetector(df)

    print_section("Detecting Numeric Columns")
    print(schema._detect_numeric_columns())

    print_section("Detecting Categorical Columns")
    print(schema._detect_categorical_columns())

    print_section("Detecting Datetime Columns")
    print(schema._detect_datetime_columns())

    print_section("Detecting ID Columns (regex safe)")
    print(schema._detect_id_columns())

    print_section("Detecting Target Column")
    print(schema._detect_target_column())


except Exception as e:
    print("\n❌ Error in schema test:")
    print(e)

