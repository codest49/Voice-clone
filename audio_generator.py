import numpy as np
from scipy.io import soundfile

def generate_audio(mfcc_features, sample_rate=16000):
    # Generate random noise to shape the MFCC features
    noise = np.random.normal(0, 0.5, size=mfccs.shape)
    
    # Combine with MFCC features
    audio_features = mfcc_features + noise
    
    # Convert to waveform (this is a simplified example)
    # In practice, you would need a neural network for this step.
    waveform = np.sin(audio_features)  # Simplified for demonstration
    
    soundfile.write("generated_audio.wav", waveform, sample_rate)
    return "generated_audio.wav"
