import os
import subprocess
import warnings
import json
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pandas as pd
import librosa
import wave
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.fft import fftshift
import noisereduce as nr
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchaudio
import torchaudio.transforms as T

with warnings.catch_warnings(): # create a context for ignoring warnings
    warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


# Visualize spectrogram of input waveform
def demo_wav(fpath):
    x, fs = librosa.load(fpath, sr=16000)
    if x.ndim != 1:
        x = x.mean(axis=1)  # Convert to mono by averaging channels
        
    print('fs', fs)
    print('x.shape', x.shape)
    
    nperseg = min(1025, len(x))
    noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(x, fs,
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   window='hann')
    print('f.shape', f.shape)
    print('t.shape', t.shape)
    print('Sxx.shape', Sxx.shape)
    plt.pcolormesh(1000*t, f/1000, 10*np.log10(Sxx/Sxx.max()),
                   vmin=-120, vmax=0, cmap='inferno')
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [ms]')
    plt.colorbar()


# perform noise reduction on input audio files and write to directory
def denoise(homedir, newdir):
    files = os.listdir(homedir)
    # Ensure the new directory exists
    if not os.path.exists(newdir):
        os.makedirs(newdir, exist_ok=True)
        
    for f in files:
        if os.path.splitext(f)[1].lower() == '.wav':
            # specify original file location
            f_loc = homedir + f    
            x, fs = librosa.load(f_loc, sr=16000)
            reduced_noise = nr.reduce_noise(y=x, sr=fs, n_fft=512)
            # specify new file location target name
            new_fname = os.path.join(newdir, f[:-4] + "_nr" + f[-4:])
            # write to new file location
            with open(new_fname, 'wb') as wf:
                wavfile.write(new_fname, fs, reduced_noise)


# Loop through list of audio files and segment them into new sample audio files for a specified keyword
def file_segment(keyword, audio_files, homedir, newdir):
    # Ensure the new directory exists
    if not os.path.exists(newdir):
        os.makedirs(newdir, exist_ok=True)
    # Initialize vosk English language model
    model = Model(lang="en-us")
    # Change keyword syntax for vosk Recognizer
    keywvox = json.dumps([keyword])
    for file in audio_files:
        if os.path.splitext(file)[1].lower() == '.wav':
            audio_filename = os.path.join(homedir, file)
            segment_fname = os.path.join(newdir, file)
            # Load the audio file with librosa
            x, fs = librosa.load(audio_filename, sr=None)
            # Convert to mono if not already
            if x.ndim != 1:
                x = librosa.to_mono(x)
            
            # Convert samples to 16-bit PCM format
            x = (x * 32767).astype(np.int16)
            # Rerite the processed audio file
            wavfile.write(audio_filename, fs, x)
            print(f"{file} converted to mono PCM format successfully.")
            # Re-open with wave
            wf = wave.open(audio_filename, "rb")
            # Create a recognizer and enable word timestamps
            rec = KaldiRecognizer(model, wf.getframerate(), f'{keywvox}')
            rec.SetWords(True)
            # Recognize speech and store the results in a list
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))
            results.append(json.loads(rec.FinalResult()))
            
            # Close the audio file
            wf.close()
            # Create a dataframe to store the word and its start and end times
            df = pd.DataFrame(columns=["word", "start", "end"])
            # Loop through the results and extract the word and its times
            print(results)
            for result in results:
                if "result" in result:
                    for word in result["result"]:
                        # Check if the word is {"keyword"}
                        if word["word"] == keyword:
                            # Create a temporary dataframe with the word and its times
                            temp_df = pd.DataFrame({"word": [word["word"]], "start": [word["start"]], "end": [word["end"]]})
                            # Concatenate the temporary dataframe with the main dataframe
                            df = pd.concat([df, temp_df], ignore_index=True)
            
            # Segment audio file and write into new .wav files
            for index, row in df.iterrows():
                sample_num = index+1
                sample_num = str(sample_num)
                audio = AudioSegment.from_wav(audio_filename)
                audio_chunk=audio[row["start"]*1000:row["end"]*1000]
                audio_chunk.export(os.path.join(segment_fname[:-4]+ "_" + sample_num + ".wav"), format="wav")
            print(str(sample_num) + " utterance(s) of keyword '" + str(keyword) + "' segmented from " + str(file) + ".")
    

# Load model and convert input sample .wav to tensor for inference
def inference_wav(model, samplewav):
    # Load audio file
    waveform, sample_rate = torchaudio.load(samplewav)
   # Get spectrogram of waveform and transform to tensor
    spectrogram_transform = T.Spectrogram()
    spectrogram = spectrogram_transform(waveform)
    print(f'Original spectrogram shape: {spectrogram.shape}')
    
    # Ensure the spectrogram contains 64 elements
    target_elements = 64
    batch_size = 1
    channels, freq_bins, time_steps = spectrogram.shape

    # Reshape or pad the tensor to the required dimensions
    if time_steps >= target_elements:
        spectrogram = spectrogram[:, :, :target_elements]
    else:
        padding = target_elements - time_steps
        spectrogram = F.pad(spectrogram, (0, padding))

    # Adjust the shape to have (batch_size, channels, freq_bins, time_steps)
    spectrogram = spectrogram.unsqueeze(0)
    
    print(f'Reshaped spectrogram shape: {spectrogram.shape}')
    
    # Initialize DSR model and inference tensor
    with torch.no_grad():
        output = model(spectrogram)
        prob = F.softmax(output, dim=1)
    return prob


# Set duration of audio file to be a certain length
def set_duration(file_path, target_duration_ms):
    audio = AudioSegment.from_file(file_path)
    if len(audio) > target_duration_ms:
        audio = audio[:target_duration_ms]
    else:
        silence = AudioSegment.silent(duration=target_duration_ms - len(audio))
        audio = audio + silence
    return audio


def process_directory(directory, target_duration_ms):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            new_audio = set_duration(file_path, target_duration_ms)
            new_audio.export(file_path, format="wav")


# Load waveforms and create ML dataset
def pad_tensor(tensor, target_length):
    # Pad the tensor to the target length
    padding = target_length - tensor.shape[-1]
    if padding > 0:
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    return tensor


# Create labeled tensor dataset from audio file directory
def create_tensor_dataset(directories, binary=True):   
    # Initialize lists
    audio_tensors = []
    filepaths = []
    for directory in directories:
        for filename in os.listdir(directory):
            try:
                file_path = os.path.join(directory, filename)
                # Load audio file
                waveform, sample_rate = torchaudio.load(filename)
                # Get spectrogram of waveform and transform to tensor
                spectrogram_transform = T.Spectrogram()
                spectrogram = spectrogram_transform(waveform)
                # Add tensor to list
                audio_tensors.append(spectrogram)
                # Add filepath to list
                filepaths.append(file_path)
            except (RuntimeError, OSError) as e:
                print(f"An error occurred with {filename}: {e}")
                continue

    if binary==True:
        # Binary class: Assign labels based on filepath name (pre & dc = 0, post = 1) Healthy = 0, Dysarthric = 1
        labels = [0 if 'pre' in s or 'dc' in s else 1 if 'post' in s else -1 for s in filepaths]
    else:
        # Multi class: Assign labels based on filepath name (pre = 0, dc = 1, post = 2)
        labels = [0 if 'pre' in s else 1 if 'dc' in s else 2 if 'post' in s else -1 for s in filepaths]
        
    # Find the maximum length of the waveforms
    max_length = max(waveform.shape[-1] for waveform in audio_tensors)
    # Pad all waveforms to the maximum length
    padded_waveforms = [pad_tensor(waveform, max_length) for waveform in audio_tensors]
    # Convert lists to tensors
    audio_tensors = torch.stack(padded_waveforms)
    labels = torch.tensor(labels)
    # Create TensorDataset
    dataset = TensorDataset(audio_tensors, labels)
    return filepaths, labels, dataset