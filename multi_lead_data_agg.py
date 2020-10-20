# data spliting method for multi lead
import os
import pandas as pd
VG_path = "/data/Viet_Gia_Clinic/D123/"
VT_path = "/data/Vien_Tim/D123_Short"

leads = ["DI", "DII", "DIII"]
all_files_with_labels = pd.DataFrame({})

for i,lead in enumerate(leads):
    file_lists = [[],[],[]]
    VT_lead_path = os.path.join(VT_path,lead + "_Short")
    VG_lead_path = os.path.join(VG_path,lead)
    file_lists[i].extend(os.listdir(VT_lead_path))
    file_lists[i].extend(os.listdir(VG_lead_path))
    # create new df for joining
    df = pd.DataFrame({'file_path_lead_'+leads[i]:file_lists[i]})
    # print(df.head(4))
    df['record_id'] = df.loc[:,'file_path_lead_'+leads[i]].str.split('_').str[:-1].agg(lambda x: '_'.join(map(str, x)))
    # join with the total df
    if (i==0):
        all_files_with_labels = df.copy()
    else:
        all_files_with_labels = pd.merge(all_files_with_labels,df,on='record_id',how='left')

print(all_files_with_labels.sample(6))
print(all_files_with_labels.shape)
all_files_with_labels.to_csv('multilead_all_labels.csv')