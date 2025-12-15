from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector
from pipeline.cleaner import DataCleaner
from pipeline.eda import EDAEngine
from pipeline.trainer import ModelTrainer
from pipeline.quality_analyzer import DatasetQualityAnalyzer


csv_path = r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv"


def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


try:
    # -----------------------------
    # Load CSV
    # -----------------------------
    loader = DataLoader(csv_path)
    df, meta = loader.load()

    print_section("CSV Loaded")
    print("Shape:", df.shape)
    print("Metadata:", meta)

    # -----------------------------
    # Schema Detection
    # -----------------------------
    schema = SchemaDetector(df).detect()

    print_section("Schema Detected")
    print(schema)

    # -----------------------------
    # Cleaning
    # -----------------------------
    cleaner = DataCleaner(df, schema)
    cleaned_df, cleaning_report = cleaner.clean()

    print_section("Cleaning Completed")
    print("Cleaned Shape:", cleaned_df.shape)
    print("Cleaning Report:", cleaning_report)

    # -----------------------------
    # EDA
    # -----------------------------
    eda_engine = EDAEngine(cleaned_df, schema)
    eda_report = eda_engine.generate_report()

    print_section("EDA Generated")
    print("EDA Keys:", list(eda_report.keys()))

    # -----------------------------
    # Training
    # -----------------------------
    trainer = ModelTrainer(cleaned_df, schema)
    prep_info = trainer.prepare_data()
    training_results = trainer.train_all_models()

    print_section("Training Results")
    print("Preparation Info:", prep_info)
    for model, scores in training_results.items():
        print(f"{model}: {scores['cv_mean_score']}")

    # -----------------------------
    # Dataset Quality Analysis
    # -----------------------------
    analyzer = DatasetQualityAnalyzer(
        schema=schema,
        eda_report=eda_report,
        training_results=training_results
    )

    quality_report = analyzer.analyze()

    print_section("DATASET QUALITY REPORT")
    for key, value in quality_report.items():
        print(f"{key}: {value}")


except Exception as e:
    print_section("ERROR")
    print(e)
