
import librosa
import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from json import JSONEncoder
#%matplotlib inline

file_name='123346__matteusnova__hello.wav'

audio_data, sampling_rate = librosa.load(file_name)
librosa.display.waveshow(audio_data,sr=sampling_rate)
ipd.Audio(file_name)

print("audio_data", audio_data)
print(len(audio_data))
print("sampling rate", sampling_rate)

audio_dataset_path='/content/'
metadata=pd.read_csv('archive/UrbanSound8K.csv')
metadata.head()

print("counts", metadata['class'].value_counts())

mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=390)

print(mfccs)
print(mfccs.shape)

def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=390)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    #file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

print("len extr feat",len(extracted_features[0]))
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(10)

print(extracted_features_df.head(10))

#with open('mfccs.txt','w') as fl:
#    json.dumps(list(mfccs))
