
clc;
clear all;

source_audio = "test_wav_files";

t = fileparts(mfilename('fullpath'))

root = fullfile(t,'..');

dirname = fullfile(root,'audio_source_files',source_audio);


len_file = fullfile(t,'..','spectrogram_data',source_audio,'clip_lengths.csv')
len_array = readtable(len_file);
num_files = height(len_array);

win = 0.5;
step = 0.5;
winMFCC = 0.05;
stepMFCC = 0.025;
visualize = 1;
before_silence_len = 10;
after_silence_len = 5;
finalpath_root = fullfile(root,'data','random_forest_results',source_audio);
mkdir(finalpath_root)

load trainedRandomForest.mat
classifier = Mdl1175;
tic
for i = 1:num_files
    fn = len_array{i,3}{1};
    id = len_array{i,2};
    fprintf('File %d of %d (%s)\n',i,num_files,fn);
    audio = string(dirname) + '\' + fn;
    finalpath = finalpath_root + "\" + pad(string(id),4,"left","0") +".xls";
    nonSpeechPrediction(audio,win,step,winMFCC,stepMFCC,classifier,before_silence_len,after_silence_len,finalpath,visualize)
    
end
toc