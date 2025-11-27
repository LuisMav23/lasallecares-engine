import os
import pandas as pd

root_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(root_folder, 'slice')
csv_files = [f for f in os.listdir(root_folder) if f.endswith('.csv')]


for f in csv_files:
    df = pd.read_csv(os.path.join(root_folder, f))
    save_name = "ASSI-A" if f == "Annual Student Screening and Interview (ASSI-A) - Alternate Form (Responses).xlsx - Form Responses 1.csv" else "ASSI-C"
    for i in range(0, len(df), 500):
        df_slice = df.iloc[i:i+500]
        df_slice.to_csv(os.path.join(root_folder, f'{os.path.splitext(f)[0]}.{i}.csv'), index=False)
