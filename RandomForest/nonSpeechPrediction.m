function nonSpeechPrediction(audio,win,step,winMFCC,stepMFCC,classifier,before_silence_len,after_silence_len,finalpath,visualize)

% INPUTS: 
%  - audio:            name of the audio signal (string)
%  - win:               window size for mid-term extraction (sec)            % 0.5 sec accepted 
%  - step:              step size for mid-term extraction (sec)              % 0.5 sec no overlap
%  - winMFCC:           window size for short term MFCC extraction (sec)     % 0.05 sec
%  - stepMFCC:          step size for short term MFCC extraction (sec)       % 0.025 half of winMFCC

% - visualize:              Flag variable to save the waveform, 1 to plot and save
% - before_silence_len:     Context before silence (in terms of units, 0.5 second is one unit)
% - after_silence_len:      Context after silence  (in terms of units, 0.5 second is one unit)
% - finalpath:                   Filename of the directory you want to save new audio recordings (string)
%  - classifier:        trained classifier                                   % e.g. Random Forest


% OUTPUT:
%  - featuresAll:       short-term features (structure)
%  - statData:          mid-term features (mean,meadian,std,min,max) 
%  - predictedLabels:   column vector of label predictions for statData
%  - resultTable:       Table including start index and time, end index and time for
%                       each silence period


% - Date:               2017-June 28
% - Created by:         Viktoria Manukyan
% - modified by:        2018-June 22


% - SAMPLE CALL:        
% [featuresAll, statData, predictedLabels, resultTable ]=nonSpeechPrediction('/Volumes/audioml/OriginalData/0221.wav',0.5,0.5,0.05,0.025,classifier,20,10,'/Volumes/audioml/Research_Latest/TESTCONTEXT',1)

%% Short-term and mid-term feature extraction
% frame by frame feature extraction 

[featuresAll,statData] =  FeatureExtractionFraming(audio,win,step,winMFCC,stepMFCC);

%% Class prediction based on given classifier

% Random Forest trained on 529 audio samples (1.7 sec-3.5 sec long)
predictedLabels = str2double(predict (classifier,statData));

[prediction,scores,stdevs] = predict(classifier,statData);




%predictedLabels = str2double(prediction);
finalpath
scores = scores(:,1);

data = [round(scores,0),scores,stdevs(:,1)];

names = ["mean","median","std","min","max"];

RF_headers = ["RF_prediction","RF_scores","RF_stdevs"];

for i = 1:5
    for ii = 1:17
        feat_headers((i-1)*17+ii) = names(i)+string(ii);
    end
end

RF_table = array2table(data,'VariableNames',RF_headers);
breakhere=1;
feat_table = array2table(statData,'VariableNames',feat_headers);

wav_name = char(audio);
wav_name = wav_name(1:4);

% path = finalpath + '/' + wav_name + '.xls';
writetable(RF_table,finalpath,'Sheet','RF_output');
writetable(feat_table,finalpath,'Sheet','MFCC_stats');
breakhere=1;

%% Generating information table for predicted silence parts
[startIndex, endIndex, resultTable, Start_Time, End_Time] = startEndPoints(predictedLabels, 4);
writetable(resultTable,finalpath,'Sheet','predicted_intervals');


breakhere=1

%timeElapsed = toc;
%% Creating wav files for silence parts which have been found
% saverecording(audio,startIndex,endIndex,predictedLabels,before_silence_len, after_silence_len,finalpath, visualize)



end
        