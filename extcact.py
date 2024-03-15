import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import librosa
import soundfile
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_path) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array(list())
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result


def load_data(data_path):
    global i
    labels = list()
    features = list()
    i = 0
    files = {'filepath': list(), 'filename': list()}
    for root, dirs, files_list in os.walk(data_path):
        for name in files_list:
            file_path = os.path.join(root, name)
            if name.endswith(".wav") and i == 0:
                print(file_path, type(file_path))
                files['filepath'].append(file_path)
                files['filename'].append(name.split('.')[0])
                i += 1

            if file_path.endswith(".wav"):
                class_label = os.path.basename(root)
                labels.append(class_label)
                feature = extract_features(file_path)
                features.append(feature)
        i = 0
    df = pd.DataFrame.from_dict(files)
    df.to_csv('labels.csv', index=False)

    return np.array(features), np.array(labels)

# Path to GTZAN dataset
data_path = "path/to/directory"

# Load data
X, y = load_data(data_path)

# Save data to CSV
df = pd.DataFrame(data=X)
df['labels'] = y
df.to_csv('gtzan_features.csv', index=False)
print("Prossess finished")


#%%
