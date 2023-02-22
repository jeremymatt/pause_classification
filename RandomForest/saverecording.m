function saverecording(audio,startIndex,endIndex,predictedLabels,before_silence_len, after_silence_len,path,visualize)



%% Function will save the selected part of audio recording

% INPUT:
% - audio:                  The original wav file name (string if not in the same directory add path)
% - startIndex:             Column vector including start points of silence
% - endIndex:               Column vector including end points of sulence
% - path:                   Filename of the directory you want to save new audio recordings (string)
% - visualize:              Flag variable to save the waveform, 1 to plot and save
% - before_silence_len:     Context before silence (in terms of units, 0.5 second is one unit)
% - after_silence_len:      Context after silence  (in terms of units, 0.5 second is one unit)
% - predictedLabels:        Predicted label vector




% - Date:           2018-June 22
% - Author:         Viktoria Manukyan



% load the audio
[signal, fs] = audioread(audio);

% define frame length
frame_len = round(fs*0.5);

% get the end of audio in terms of units
veryend = length(predictedLabels);

for silence=1:length(startIndex)
    if  (startIndex(silence)- before_silence_len) > 0 && (endIndex(silence)-after_silence_len) < veryend
        newsignal = signal((startIndex(silence)-1)*frame_len:endIndex(silence)*frame_len);
        filename = [audio(end-7:end-4) '_' num2str(startIndex(silence)) '_' num2str(endIndex(silence)) '.wav'];
        name = [path '/' filename];
        
        audiowrite(name,newsignal,fs)
        if visualize == 1
            t=(1/fs:1/fs:length(newsignal)/fs);
            h = plot(t,newsignal);
            saveas(h,[name '.png'])
        end
    elseif startIndex(silence)- before_silence_len <=0
        newsignal = signal(1:endIndex(silence)*frame_len);
        filename = [audio(end-7:end-4) '_' num2str(startIndex(silence)) '_' num2str(endIndex(silence)) '.wav'];
        name = [path '/' filename];
        
        audiowrite(name,newsignal,fs)
        if visualize == 1
            t=(1/fs:1/fs:length(newsignal)/fs);
            h = plot(t,newsignal);
            saveas(h,[name '.png'])
        end
        
    elseif endIndex(silence)- after_silence_len >= veryend
        signal((startIndex(silence)-1)*frame_len:veryend);
        filename = [audio(end-7:end-4) '_' num2str(startIndex(silence)) '_' num2str(endIndex(silence)) '.wav'];
        name = [path '/' filename];
        audiowrite(name,newsignal,fs)
        if visualize == 1
            t=(1/fs:1/fs:length(newsignal)/fs);
            h = plot(t,newsignal);
            saveas(h,[name '.png'])
        end
          
        
    end
end





