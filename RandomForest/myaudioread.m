function [signal, fs] = myaudioread(audio)
% Function to convert STEREO TO MONO

% INPUT:
%  - audio:         name of the audio signal (string)

% OUTPUT:
%  - signal:        converted MONO signal

%  - Date:           April 12, 2017

%  - Created by:     Viktoria Manukyan

[signal, fs] = audioread(audio);

if (size(signal,2)>1)
    signal = (sum(signal,2)/2); % convert to MONO
end

