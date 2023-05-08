# Music-Genre-Classification
 Feature extraction and genre classification for the GTZAN dataset

Project developed jointly with Kseniya Ruta for Georgia Tech graduate course ISYE 6740 Computational Data Analytics

Key files:
- MusicGenreClassification.ipynb: Jupyter notebook with code to extract features from raw audio files,
   train various classifier models, and evaluate performance against genre labels.
- MusicUtilities.py: utility functions for data & feature I/O plus various feature extraction algorithms
- DOE_matrix.csv: Design of Experiments grid assigning various feature-set combinations to different runs of the program
- features\\ subfolder: extracted features for GTZAN clips
- features2\\ subfolder: extracted features for validation audio clips

To run the feature extraction, you'll need the GTZAN dataset itself:
https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

And you should probably read the Tzanetakis paper:
Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302

Validation files are optional user-provided files to test the classifiers
- these should be in a subfolder Genre_Samples_Converted\\ with folders as the name of each genre (matching the GTZAN names) i.e. "Blues"
