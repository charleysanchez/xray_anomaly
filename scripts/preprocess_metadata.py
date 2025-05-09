import pandas as pd

csv_path = 'data/xray_images/Data_Entry_2017.csv'

df = pd.read_csv(csv_path)

df = df[df['View Position'] == 'PA'].copy()

all_labels = set()
for labels in df['Finding Labels']:
    for label in labels.split('|'):
        all_labels.add(label.strip())

all_labels = sorted(all_labels)

for label in all_labels:
    df[label] = df['Finding Labels'].apply(lambda x: int(label in x.split('|')))


print("Class distribution:")
print(df[all_labels].sum().sort_values(ascending=False))

df.to_csv('data/xray_images/preprocessed_metadata.csv', index=False)