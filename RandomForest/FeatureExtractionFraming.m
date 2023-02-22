function  [featuresAll, stat_features] =  FeatureExtractionFraming(signal,win,step,winMFCC,stepMFCC)

%% mid-term MFCC frame by frame feature extraction

% Author: Viktoria Manukyan
% Date: 2017-04-01

% INPUTS: 
%  - signal:            name of the audio signal (string)
%  - win:               window size for mid-term extraction (sec)            % 2 sec accepted for MFCC
%  - step:              step size for mid-term extraction (sec)              % 90% overlap accepted
%  - winMFCC:           window size for short term MFCC extraction (sec)     % 0.02-0.1sec
%  - stepMFCC:          step size for short term MFCC extraction (sec)       % half of winMFCC
% OUTPUT:
%  - featuresAll:       short-term features (structure)
%  - stat_features:     mid-term features (mean,meadian,std,min,max) 
%  - timeElapsed:       feature extraction time    

%  - Date:      April 12, 2017
%  - Created by:    Viktoria Manukyan


%  - Sample:            FeatureExtractionFraming('test4.wav',0.5,0.5,0.05,0.025);

[signal,fs] = myaudioread(signal);
%%%%% only for deep learning to get first 8 second of audio recording
if false
    if fs == 16000
        signal = signal(1:1.4008e+05);
    elseif fs == 441000
        signal = signal(1:3.8610e+05);
    end
end
%if size(signal,1) < 3.8610e+05
%signal  = signal(1:1.4008e+05);
%end


% convert window length and step from seconds to samples:
windowLength = round(win * fs);
step_length = round(step * fs);

%compute the length of the signal in samples
L = length(signal);

% compute the total number of frames:
numOfFrames = floor((L-windowLength)/step_length) + 1;

% for each frame compute the short term MFCC features
for k = 1:numOfFrames
    %frame = signal((k-1)*windowLength+1: windowLength*k);
    frame = signal((k-1)*step_length+1:(k-1)*step_length+windowLength);
    featuresAll(k).features = stFeatureExtraction(frame, fs, winMFCC, stepMFCC);
end


% number of short term features
numfeatures = 17; 
% number of mid term statistic features
numstats = 5;

% preallocate the mid-term features
stat_features = zeros(numOfFrames,numfeatures*numstats);


for j = 1:size(stat_features,1)
    stat_features(j,(1:17)*numstats-(numstats-1)) = mean  (featuresAll(j).features, 2); % compute the mean for all MFCC features (particular frame j)
    stat_features(j,(1:17)*numstats-(numstats-2)) = median(featuresAll(j).features, 2); % compute the median for all MFCC features (particular frame j)
    stat_features(j,(1:17)*numstats-(numstats-3)) = std   (featuresAll(j).features,0,2); % compute the std for all MFCC features (particular frame j)
    stat_features(j,(1:17)*numstats-(numstats-4)) = min   (featuresAll(j).features,[],2); % compute the min for all MFCC features (particular frame j)
    stat_features(j,(1:17)*numstats-(numstats-5)) = max   (featuresAll(j).features,[],2); % compute the max for all MFCC features (particular frame j)
end





