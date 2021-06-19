import pandas as pd

df = pd.read_csv('captions.csv')
df = df[['image_name', 'caption']].groupby('image_name')['caption'].apply(list).reset_index(name='sentences')
df.drop_duplicates(subset=['image_name'], inplace=True)

flist = []
count = 0
for i, row in df.iterrows():
    count += 1
    temp = {}
    temp['sentences'] = row['sentences']
    temp['id'] = int(count)
    temp['img_path'] = str(row['image_name'])

    flist.append(temp)

with jsonlines.open('flickr30k.jsonline', 'w') as writer:
    writer.write_all(flist)