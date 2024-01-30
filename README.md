# Classically punk

## Task


<div class="row">
<div class="col tab-content">
<div class="tab-pane active show" id="subject" role="tabpanel">
<div class="row">
<div class="col-md-12 col-xl-12">
<div class="markdown-body">
<p class="text-muted m-b-15">
</p><h2>Classically Punk</h2>
<table>
<thead>
<tr>
<th>Technical details</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Submit file</td>
<td>*.ipynb</td>
</tr>
</tbody>
</table>
<hr>
<h1>Classically Punk</h1>
<p>Music. Experts have been trying to understand sound and what differentiates one song from another for a long time. What makes a tone different from another? Can we visualize sound and if yes, how?
Audio files contain a lot of data, and our goal is to understand what an audio file is, the data it contains, and what we can do with that data. What features can we visualize on this kind of data?</p>
<p><em>Your Mission</em></p>
<p>Your first mission will be to find a library that "reads" music. (Computers don`t listen to music, they read it :) )
Then your second mission will be to separate a collection of music files into different music genres, and to do so, you will need to be able to first identify features inside the music files that will be used for classification.</p>
<p>What to expect:
Build an application in Python that automatically classifies different musical genres from an audio snippet.
You should expect to handle large data sets. You should also expect to analyse media files to generate data and identify patterns.</p>
<p>Your deliverables:</p>
<ul>
<li>A presentation with slides on how you classified the music, as well as assumptions, implications, and other important information.</li>
<li>Code that the DevOps team should be able to push to production.</li>
</ul>
<h2>Technical specifications</h2>
<p>Implement a multi-variable linear regression model on a large and complex data set.
Analyze and evaluate the implications of your model on real-life users.
Analyze and evaluate the risk to the business of the impact, assumptions, and decisions in your model</p>
<p>Learning outcomes: the five stages of your project
In this project, you should expect to cover the five major stages of working with data:</p>
<ol>
<li>Data Collecting / Cleaning (see below)</li>
<li>Data Exploration</li>
<li>Data Visualization</li>
<li>Machine Learning</li>
<li>Communication</li>
</ol>
<p>You will have to prove yourself in each of these. We are confident that you will succeed! :)</p>
<p>Where to find the data?</p>
<ul>
<li>
<a href="https://storage.googleapis.com/qwasar-public/track-ds/classically_punk_music_genres.tar.gz" target="_blank">Music Data Set ~1.2go</a>
-&gt; If you upload this data set inside Docode, it will freeze it.
Work directly from your computer for this project.</li>
</ul>
<p>This dataset was used for the well-known paper in genre classification "Musical genre classification of audio signals" by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.</p>
<p>Reminder, it will be one of your portfolio projects. You can find a lot of different ideas. Plagiarism is not tolerated in the company either here. :-)</p>

<p></p>
</div>

</div>
</div>
</div>
<div class="tab-pane" id="resources" role="tabpanel">
<div class="row">
<div class="col-xl-12">
<div class="row text-center">
<div class="col">
<a target="_blank" href="https://blog.clairvoyantsoft.com/music-genre-classification-using-cnn-ef9461553726">https://blog.clairvoyantsoft.com/music-genre-classification-using-cnn-ef9461553726</a>
</div>
</div>
<hr>
<div class="row text-center">
<div class="col">
<a target="_blank" href="https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8">https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8</a>
</div>
</div>
<hr>
<div class="row text-center">
<div class="col p-t-10 f-12">
<p>
How To Use Jupyter In Docode
</p>
</div>
</div>
<div class="row text-center">
<div class="col">
<a frameborder="0" href="https://www.youtube.com/embed/J5MpsvScKzE">How to use jupyter in docode</a>
</div>
</div>

</div>
</div>
</div>
</div>
</div>


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
   git clone https://github.com/newjasjanpython/03-Machine-Learning-Classically-Punk-.git
   cd 03-Machine-Learning-Classically-Punk-
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
