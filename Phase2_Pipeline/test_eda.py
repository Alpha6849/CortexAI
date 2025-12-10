from pipeline.loader import DataLoader
from pipeline.schema import SchemaDetector
from pipeline.cleaner import DataCleaner
from pipeline.eda import EDAEngine
import pprint

print("\n=== Running EDA Basic Stats Test ===\n")

loader = DataLoader(r"D:\AI scientist\Phase1_AutoML_EDA\data\Sample.csv")
df, meta = loader.load()

schema = SchemaDetector(df).detect()
cleaned_df, report = DataCleaner(df, schema).clean()

eda = EDAEngine(cleaned_df, schema)
stats = eda.generate_basic_statistics()

pprint.pprint(stats)
