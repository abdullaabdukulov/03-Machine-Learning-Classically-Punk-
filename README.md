# Classically punk

## Task

## Description

This project explores the world of music classification through data science, aiming to unravel the secrets of music by leveraging a renowned dataset for musical genre classification. The core of the project involves creating a Python application that automatically classifies music genres based on audio snippets. The process includes essential tasks such as downloading the dataset, extracting features from audio frames, visualizing waveforms, creating spectrograms, and building a genre classification model using TensorFlow.

## Example of Code

### Downloading and Extracting Dataset

```python
if not os.path.exists('genres'):
    with open('genres_tar_file.tar.gz', 'wb') as file:
        response = requests.get('https://storage.googleapis.com/qwasar-public/track-ds/classically_punk_music_genres.tar.gz')
        file.write(response.content)

    with tarfile.open('genres_tar_file.tar.gz', 'r:gz') as tar:
        members = tar.getmembers()

        for member in members:
            if member.name.startswith(f'genres/'):
                tar.extract(member, path=os.getcwd())
```

### Feature Extraction using AudioFeatureExtractor

```python
extractor = AudioFeatureExtractor('genres')
extractor.process()
df = extractor.save_csv('genres.csv')
print(df.shape)
print(df)
```

### Plotting Waveforms

```python
plot_waveforms("genres")
```

### Model Creation and Training

```python
from model_utils import create_and_train_model

X_train_scaled, X_validation_scaled, X_test_scaled, y_train_encoded, y_validation_encoded, y_test_encoded = preprocess_data(df)

model = create_and_train_model(X_train_scaled, y_train_encoded, X_validation_scaled, y_validation_encoded)
```

## Installation

To run this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas tensorflow scikit-learn matplotlib seaborn
   ```

3. Run the main script:

   ```bash
   ipython classically-punk.ipynb
   ```

## Usage

After following the installation steps, you can use the provided scripts and functions to explore the dataset, extract audio features, visualize waveforms, and train the genre classification model. Customize parameters and experiment with hyperparameters to optimize the model's performance.

## Summary

In summary, this project delves into the fascinating realm of music classification through data science. Leveraging a robust dataset and a combination of libraries and frameworks, we've created a Python application capable of automatically classifying music genres based on audio snippets. The project encompasses feature extraction, waveform visualization, spectrogram exploration, and model creation, providing a comprehensive journey into the harmonies and nuances of sound. Further experimentation with hyperparameters is encouraged to enhance the model's accuracy, and this project serves as a foundation for future advancements in audio signal processing and machine learning in music classification.
