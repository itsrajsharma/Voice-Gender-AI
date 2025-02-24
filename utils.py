import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13, sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
