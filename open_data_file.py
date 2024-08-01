import pandas as pd


file = "/home/arr65/data/nshm/nshm_data/nloc_0=-34.0~173.0/7773a233-d86e-41d5-8057-1bc86b3e26b9-part-0.parquet"
df = pd.read_parquet(file)

print()