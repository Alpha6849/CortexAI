from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector
from pipeline.cleaner import DataCleaner
from pipeline.eda import EDAEngine
from pipeline.trainer import ModelTrainer
import pprint

print("\n=== Running Trainer Preparation Test ===\n")

loader = DataLoader(r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv")
df, meta = loader.load()

# Schema
schema = SchemaDetector(df).detect()

# Clean
cleaned_df, clean_report = DataCleaner(df, schema).clean()

# Init Trainer
trainer = ModelTrainer(cleaned_df, schema)

# Prepare data
prep = trainer.prepare_data()
print("\nPrepare Data Output:")
pprint.pprint(prep)
