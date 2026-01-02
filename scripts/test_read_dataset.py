from pathlib import Path
import pandas as pd

file_name = 'Data file for students.xlsx'
found = None
for p in [Path.cwd()] + list(Path.cwd().parents):
    candidate = p / 'datasets' / file_name
    if candidate.exists():
        found = candidate
        break
print('found:', found)
if not found:
    raise SystemExit('not found')

df = pd.read_excel(found, engine='openpyxl')
print('shape:', df.shape)
print(df.head(2))
