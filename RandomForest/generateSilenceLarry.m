
% Script loads predictions for each recording and generates wav files for
% each predicted silence instance in 2 forms - with and without context

% Viktoria Manukyan 
% Edited: Larry Clarfeld (edited paths and "save" function input 7/27/18)

% getFilenamesFromFolders
% clearvars -except files_all

p = pwd;

% For PC Users:
path = [p '\Samples27000C'];
addpath([p '\extractedData']);
addpath([p(1:3) 'OriginalData'])
filenames = dir([p '\extractedData']);

% For Mac Users:
% path = '/Volumes/audioml/Research_Latest/Samples27000C';
% addpath '/Volumes/audioml/Research_Latest/extractedData'
% addpath '/Volumes/audioml/OriginalData'
% filenames = dir('/Volumes/audioml/Research_Latest/extractedData');

n_silences = zeros(size(filenames,1),1);

low_qual_recs = [160 184 219 244 266 607]; 

for i = 1:length(filenames)
    filename = filenames(i).name;
    if filename(1) == '0'
        recnum = str2double(filename(1:end-4));
        if length(conv_cts(conv_cts(:,1)==recnum,2)) == 1 && ...
                ~ismember(recnum, low_qual_recs)
            if recnum > 212 && recnum < 219
                load(filename)
                n_silences(i) = numel(startIndex);
                audio = [filename(1:end-4),'.wav'];
                [signal, fs] = audioread(audio);
        %         predictedSilence(startIndex, endIndex, ...
        %         audio,predictedLabels,0.5,path,signal,fs); % obselete function
                visualize = 0;
                before_silence_len = 10.*2; % 10 seconds context before
                after_silence_len = 5.*2; % 5 seconds context after

                saverecordingLarry(audio,startIndex,endIndex,predictedLabels,before_silence_len, after_silence_len,path,visualize) 
                disp(['File ' filename ' complete'])
            end
        end
    end
end