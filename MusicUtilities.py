# Music Analysis Project Utilities

import librosa
import matplotlib.pyplot as plt
import librosa.display
from statistics import mean
import numpy as np
import scipy.stats
import scipy.signal
import pandas as pd
from scipy.stats import skew, kurtosis
import os
from sklearn import svm, metrics
import sklearn
from scipy.signal import butter,filtfilt

def get_mfcc_features(x,sr,n_mfcc=12,verbose=False):
  # Extract Mel-frequency Cepstral Coefficient (MFCC)
  mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=12)
  mfccs_dB = librosa.amplitude_to_db(mfccs)
  mfccs_mean = np.mean(mfccs_dB, axis=1)
  mfccs_std = np.std(mfccs_dB, axis=1)
  mfccs_skewness = skew(mfccs_dB, axis=1) 
  mfccs_kurtosis = kurtosis(mfccs_dB, axis=1)
  
  # Calculate covariance matrix
  cov_matrix = np.cov(mfccs_dB)        
  # Extract upper triangular matrix
  triangular = np.triu(cov_matrix, k=1)
  # Get the indices of the upper triangular matrix
  iu = np.triu_indices(cov_matrix.shape[0], k=1)
  # Flatten triangular matrix to 1D array with all non-zero elements
  triangular_1d = triangular[iu]

  mfcc_features = np.concatenate((mfccs_mean, mfccs_std, mfccs_skewness, 
                                  mfccs_kurtosis, triangular_1d))
  return mfcc_features


def get_ecfz_features(x,sr,verbose=False):
  # Extract Energy
  energy = librosa.feature.rms(y=x)
  energy_mean = energy.mean()
  energy_std = energy.std()
  energy_skew = skew(energy.flatten())
  energy_kurt = kurtosis(energy.flatten())

  # Extract spectral centroid
  spec_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
  spec_centroid_mean = np.mean(spec_centroid[0])
  spec_centroid_std = np.std(spec_centroid[0])
  spec_centroid_skew = skew(spec_centroid[0])
  spec_centroid_kurt = kurtosis(spec_centroid[0])

  # Extract Flux
  flux = librosa.onset.onset_strength(y=x, sr=sr)
  flux_mean = flux.mean()
  flux_std = flux.std()
  flux_skew = skew(flux.flatten())
  flux_kurt = kurtosis(flux.flatten())

  # Extract Zero-crossing rate
  zcr = librosa.feature.zero_crossing_rate(y=x)
  zcr_mean = np.mean(zcr)
  zcr_std = np.std(zcr)
  zcr_skew = skew(zcr.flatten())
  zcr_kurt = kurtosis(zcr.flatten())

  # Extract bandwidth
  bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
  bandwidth_mean = np.mean(bandwidth[0])
  bandwidth_std = np.std(bandwidth[0])
  bandwidth_skewness = skew(bandwidth[0])
  bandwidth_kurtosis = kurtosis(bandwidth[0])

  # Extract rolloff85
  rolloff85 = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
  rolloff85_mean = np.mean(rolloff85[0])
  rolloff85_std = np.std(rolloff85[0])
  rolloff85_skewness = skew(rolloff85[0])
  rolloff85_kurtosis = kurtosis(rolloff85[0])

  # Extract rolloff95
  rolloff95 = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.95)
  rolloff95_mean = np.mean(rolloff95[0])
  rolloff95_std = np.std(rolloff95[0])
  rolloff95_skewness = skew(rolloff95[0])
  rolloff95_kurtosis = kurtosis(rolloff95[0])

  # Extract flatness
  flatness = librosa.feature.spectral_flatness(y=x)
  flatness_mean = np.mean(flatness)
  flatness_std = np.std(flatness)
  flatness_skewness = skew(flatness.flatten())
  flatness_kurtosis = kurtosis(flatness.flatten())

  # Extract contrast
  contrast_spread = librosa.feature.spectral_contrast(y=x, sr=sr)
  contrast_mean = np.mean(contrast_spread[0])
  contrast_std = np.std(contrast_spread[0])
  contrast_skewness = skew(contrast_spread[0])
  contrast_kurtosis = kurtosis(contrast_spread[0])

  if verbose:
    print(energy_mean,spec_centroid_mean,flux_mean,zcr_mean)
    print(bandwidth_mean,rolloff85_mean,rolloff95_mean,flatness_mean,contrast_mean)

  ecfz_features = [energy_mean, energy_std, energy_skew, energy_kurt,
    spec_centroid_mean, spec_centroid_std,
    spec_centroid_skew, spec_centroid_kurt,
    flux_mean, flux_std, flux_skew, flux_kurt,
    zcr_mean, zcr_std, zcr_skew, zcr_kurt,
    bandwidth_mean, bandwidth_std, bandwidth_skewness, bandwidth_kurtosis,
    rolloff85_mean, rolloff85_std, rolloff85_skewness, rolloff85_kurtosis,
    rolloff95_mean, rolloff95_std, rolloff95_skewness, rolloff95_kurtosis,
    flatness_mean, flatness_std, flatness_skewness, flatness_kurtosis,
    contrast_mean, contrast_std, contrast_skewness, contrast_kurtosis]

  return ecfz_features

def get_beat_features(x,sr,verbose=False):
  # Function to extract beat histogram features from audio clip
  # Following the Tzanetakis 2002 paper method
  # returns an array of features, as follows:
  #   amp_sum = sum of beat histogram energy
  #   a0, a1 = relative amplitude of first 2 peaks / amp_sum
  #   ra = a1/a0 = relative amplitude of peak 2 / peak 1
  #   p0, p1 = period in bpm of first 2 peaks
  # verbose flag will print several charts and final metrics
  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Original waveform")
    plt.plot(x[0:10500])

  # 1. Full Wave Rectification
  x_rec = np.abs(x)
  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 1 - Rectification")
    plt.plot(x_rec[0:10500])

  # 2. Low-Pass Filter
  nn = x_rec.shape[0]
  x_lp = np.zeros(nn)
  alpha = 0.9
  #x_lp[0] = x_rec[0]
  #x_lp[1:nn] = x_rec[1:nn]*(1-alpha) + x_rec[0:nn-1]*alpha
  b, a = butter(N=4,Wn=250,btype='low',fs=sr)
  x_lp = filtfilt(b, a, x_rec)

  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 2 - Low Pass Filter")
    plt.plot(x_rec[0:10500])
    plt.plot(x_lp[0:10500])

  # 3. Downsampling
  ds_factor = 15  # pick even multiple of 22050
  x_ds = librosa.resample(y=x_lp, orig_sr=sr,target_sr=sr//ds_factor)

  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 3 - Downsampled")
    plt.plot(x_ds[0:700])

  # 4. Mean Removal
  #x_mr = x_ds
  x_mr = x_ds - np.mean(x_ds)

  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 4 - Mean removal")
    plt.plot(x_mr[0:700])

  # 5. Enhanced Autocorrelation
  x_ac = np.zeros(2500)
  nn=x_mr.shape[0]
  # 5a - perform autocorrelation
  for k in range(0,2500):   # roughly 200-40 bpm at downsampled 1470 Hz
    N = nn-k
    x_ac[k] = np.sum(x_mr[0:N]*x_mr[k:nn])/N
  x_acn = librosa.util.normalize(x_ac)
  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 5a - Autocorrelation")
    plt.plot(range(x_acn.shape[0]),x_acn)

  # 5b - subtract higher orders, clip to min zero
  min_clip = 0
  nn=x_acn.shape[0]
  x_acn2 = np.clip(x_acn,a_min=min_clip,a_max=None)
  for n in range(2,5):
    inds = np.arange(0,nn,n)
    ac_global2  = np.repeat(np.nan,nn)
    for ind in inds:
        ac_global2[ind] = x_acn[ind//n]

    ac_global3 = np.clip(np.array(pd.DataFrame(ac_global2).interpolate()).flatten(),a_min=min_clip,a_max=None)
    x_acn2 = np.clip(x_acn2 - ac_global3,a_min=0,a_max=None)

  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 5b - Enhanced Autocorrelation")
    plt.plot(range(x_acn2.shape[0]),x_acn2)

  # 5c - detect AC peaks
  acpeaks = scipy.signal.find_peaks(x_acn2,distance=20)

  # 5c - convert to tempo in 1bpm increments
  tempos = range(50,200,1)
  nt = len(tempos)
  bh = np.zeros((nt,2))
  bh[:,0] = tempos
  '''
  # alternate approach - just add AC peaks to tempo histogram
  for peak in acpeaks[0]:
    t = np.round(sr/ds_factor/peak*60)
    bh[int(t)-50,1] = x_acn2[peak]
  '''
  # previous approach
  for i in range(nt):
    t = tempos[i]
    lag = round(sr/ds_factor*60/t)
    acind = np.argmax(x_acn2[lag-4:lag+5])
    bh[i,1] = x_acn[lag-4+acind]

  bh = np.clip(bh,a_min=0,a_max=None)

  if verbose:
    plt.figure(figsize=(15, 3))
    plt.title("Step 5c - Beat Histogram")
    plt.plot(bh[:,0],bh[:,1])

  # 6. Peak Detection & Histogram calculation
  peakinds = scipy.signal.find_peaks(bh[:,1],distance=20)
  bh_peaks = bh[peakinds[0],:]
  bh_peaks = bh_peaks[np.argsort(-bh_peaks[:, 1])]
  if bh_peaks.shape[0] == 1:
    bhp2 = np.zeros((2,2))
    bhp2[0,:] = bh_peaks[0,:]
    bh_peaks = bhp2.copy()

  # 7. Metric Calculations
  amp_sum = np.sum(bh_peaks[:,1])
  a0 = bh_peaks[0,1]/amp_sum
  a1 = bh_peaks[1,1]/amp_sum
  ra = a1/a0
  p0 = bh_peaks[0,0]
  p1 = bh_peaks[1,0]

  # 8. Return features
  beat_features = [a0,a1,ra,p0,p1,amp_sum]
  if verbose:
    print(beat_features)

  return beat_features

# function to take an array of 12 durations
#   and shift so the max element is index zero
#   with wrapping lower-index elements to the end
#   for example: abcdEfghijkl --> Efghiklabcde
def recenter_12tones(inputdf,verbose=False):
  ctr_df = np.zeros(12)
  key = np.argmax(inputdf)

  ctr_df[0:(12-key)] = inputdf[key:12].copy()
  ctr_df[(12-key):12] = inputdf[0:key].copy()
  
  return ctr_df, key

# function accepts an audio array and sampling rate
#   and returns 12 features describing the relative occurrence
#   of the 12 scale tones
#   with the dominant tone shifted to the 1st position (index 0)
def get_stp_features(audio_clip, sr, verbose = False):

  tones = ["A","Bb","B","C","Db","D","Eb","E","F","Gb","G","Ab"]
  scaletones = ["1","b2","2","m3","M3","4","b5","5","b6","6","m7","7"]

  # split to harmonic elements only
  audio_h, audio_p = librosa.effects.hpss(audio_clip)
  if verbose:
    print(f"Shape of harmonic component: {audio_h.shape}")

  # extract chromagram array of scale tones over time
  hl = 512
  chromagram = librosa.feature.chroma_stft(y=audio_h, sr=sr, hop_length=hl)
  if verbose:
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hl, cmap='coolwarm')

  # perform longitudinal sum and normalize
  csum = np.sum(chromagram,axis=None)
  rsum = np.sum(chromagram,axis=1) / csum
  if verbose:
    print("Shape of rowsum:",rsum.shape)
    print("Rowsums:",np.round(rsum,3))

    fig1, ax1 = plt.subplots()
    ax1.bar(tones,rsum)
    ax1.set_title("Occurrence Frequency of Notes")
    ax1.set_xlabel("Named note")
    ax1.set_ylabel("Relative Frequency")
    plt.show()

  # recenter so most prevalent note is index 0
  stp, key = recenter_12tones(rsum)
  if verbose:
    print("Shape of stp:",rsum.shape)
    print("Detected key:",key)
    print("STP:",np.round(stp,3))

    fig2,ax2 = plt.subplots()
    ax2.bar(scaletones,stp)
    ax2.set_title("Harmonized Occurrence of Notes")
    ax2.set_xlabel("Scale tone")
    ax2.set_ylabel("Relative Frequency")
    plt.show()

  stp = np.append(stp,key)

  return stp

# function to extract requested features
#   checks for presence of feature CSV files in 'features' subfolder
# returns an array of filenames, 
#   and an numpy array of songs (rows) and features (columns)
# function expects features to be in \\features subfolder of 'path'
#   and audio files to be in \\GTZAN_data\\<genrename> subfolder of 'path'
def get_music_features(featuredict,verbose=True):

  feat_array = np.array(range(999)).reshape((999,1))

  # for each requested feature
  for f in featuredict.keys():
    # else check if requested feature files exist  
    #targetfile = path+'\\features\\'+str(f)+'_features.csv'
    targetfile = 'features\\'+str(f)+'_features.csv'
    # load available features
    if os.path.isfile(targetfile) and featuredict[f]==True:
      # load features into numpy array
      feat_data = pd.read_csv(targetfile)
      feat_data = feat_data.iloc[:,1:]
      feat_array = np.concatenate((feat_array,np.array(feat_data)),axis=1)
    else:
      if os.path.isfile(targetfile) == False:
        print("Warning: feature file for feature",f,"not found")
      else:
        print("Skipping feature",f,"as requested")
  
  return feat_array[:,1:]

def get_music_features2(featuredict,verbose=True):

  feat_array = np.array(range(28)).reshape((28,1))

  # for each requested feature
  for f in featuredict.keys():
    # else check if requested feature files exist  
    #targetfile = path+'\\features2\\'+str(f)+'_features.csv'
    targetfile = 'features2\\'+str(f)+'_features.csv'
    # load available features
    if os.path.isfile(targetfile) and featuredict[f]==True:
      # load features into numpy array
      feat_data = pd.read_csv(targetfile)
      feat_data = feat_data.iloc[:,1:]
      feat_array = np.concatenate((feat_array,np.array(feat_data)),axis=1)
    else:
      if os.path.isfile(targetfile) == False:
        print("Warning: feature file for feature",f,"not found")
      else:
        print("Skipping feature",f,"as requested")
  
  return feat_array[:,1:]

# function to extract all requested features and save to local files
# 'featuredict' should be a dictionary of feature names
#   with a boolean for whether to recalculate each
#   function expects audio files to be in \\GTZAN_data\\<genrename> subfolder of 'path'
#   function saves features in \\features subfolder of 'path'
def create_feature_files(featuredict,verbose=True):

  genres = ['blues','classical','country','disco',
            'hiphop','jazz','metal','pop',
            'reggae','rock']
  poss_feats = ['MFCC','ECFZ','BeatHist','STP']

  # initialize feature arrays
  mfcc_array = np.zeros((999,114))
  ecfz_array = np.zeros((999,36))
  bf_array = np.zeros((999,6))
  stp_array = np.zeros((999,13))  

  filenames = []
  j = 0

  # loop through all the songs and extract requested features
  for gi in range(10):
    g = genres[gi]
    if verbose:
      print("Processing genre:",g)
    for i in range(100):
      # construct expected song name
      if i<10:
        songfile = g+'.0000'+str(i)
      else:
        songfile = g+'.000'+str(i)
      #audio_path = path+'\\GTZAN_data\\'+g+'\\'+songfile+'.wav'
      audio_path = 'GTZAN_data\\'+g+'\\'+songfile+'.wav'
      # if song file exists, extract features
      if os.path.isfile(audio_path):
        x, sr = librosa.load(audio_path)

        if featuredict['MFCC']:
          mfcc_array[j,:] = get_mfcc_features(x,sr,n_mfcc=12,verbose=verbose)
        if featuredict['ECFZ']:
          ecfz_array[j,:] = get_ecfz_features(x,sr,verbose=verbose)
        if featuredict['BeatHist']:
          bf_array[j,:] = get_beat_features(x, sr)
        if featuredict['STP']:
          stp_array[j,:] = get_stp_features(x, sr)

        filenames.append(songfile)
        j += 1
  
  filesaved = []

  # save individual feature set files
  if featuredict['MFCC']:
    df = pd.DataFrame(mfcc_array)
    df.to_csv('features\\MFCC_features.csv',sep=',')
  if featuredict['ECFZ']:
    df = pd.DataFrame(ecfz_array)
    df.to_csv('features\\ECFZ_features.csv',sep=',')
  if featuredict['BeatHist']:
    df = pd.DataFrame(bf_array)
    df.to_csv('features\\BeatHist_features.csv',sep=',')
  if featuredict['STP']:
    df = pd.DataFrame(stp_array)
    df.to_csv('features\\STP_features.csv',sep=',')

  return filesaved


# function to extract all requested features and save to local files
# 'featuredict' should be a dictionary of feature names
#   with a boolean for whether to recalculate each
#   function expects audio files to be in \\GTZAN_data\\<genrename> subfolder of 'path'
#   function saves features in \\features subfolder of 'path'
def create_feature_files2(filelist,featuredict,verbose=True):

  nfiles = len(filelist)

  # initialize feature arrays
  mfcc_array = np.zeros((nfiles,114))
  ecfz_array = np.zeros((nfiles,36))
  bf_array = np.zeros((nfiles,6))
  stp_array = np.zeros((nfiles,13))  

  filenames = []
  j = 0

  # loop through all the songs and extract requested features
  for file in filelist:
    pathsplit = file.split("\\")
    songfile = pathsplit[len(pathsplit)-1]
    g = pathsplit[len(pathsplit)-2]

    if verbose:
      print("Processing file:",g,songfile)
      # if song file exists, extract features
      if os.path.isfile(file):
        x, sr = librosa.load(file)

        if featuredict['MFCC']:
          mfcc_array[j,:] = get_mfcc_features(x,sr,n_mfcc=12,verbose=verbose)
        if featuredict['ECFZ']:
          ecfz_array[j,:] = get_ecfz_features(x,sr,verbose=verbose)
        if featuredict['BeatHist']:
          bf_array[j,:] = get_beat_features(x, sr)
        if featuredict['STP']:
          stp_array[j,:] = get_stp_features(x, sr)

        j += 1
  
  filesaved = []

  # save individual feature set files
  if featuredict['MFCC']:
    df = pd.DataFrame(mfcc_array)
    df.to_csv('features2\\MFCC_features.csv',sep=',')
  if featuredict['ECFZ']:
    df = pd.DataFrame(ecfz_array)
    df.to_csv('features2\\ECFZ_features.csv',sep=',')
  if featuredict['BeatHist']:
    df = pd.DataFrame(bf_array)
    df.to_csv('features2\\BeatHist_features.csv',sep=',')
  if featuredict['STP']:
    df = pd.DataFrame(stp_array)
    df.to_csv('features2\\STP_features.csv',sep=',')

  return filesaved

# function to run selected classifier model against the data
def run_classification(modelname,cdict,data_array,verbose=False):

  # Extract datasets from array
  Xtrain, Xtest, Ytrain, Ytest = data_array

  # train and predict with the selected model
  clf = cdict[modelname]
  #print(method)
  clf.fit(Xtrain,Ytrain)
  predicted = clf.predict(Xtest)
  testing_accuracy = clf.score(Xtest, Ytest)
  if verbose:
    #print("Classification report for classifier " + modelname + ":")
    #print(metrics.classification_report(Ytest, predicted))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytest, predicted))

    print("Testing accuracy of", modelname, testing_accuracy)
    print("#######################################################")

  return clf, predicted, testing_accuracy
