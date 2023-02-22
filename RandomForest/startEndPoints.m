function [startIndex, endIndex, resultsTable, Start_Time, End_Time] = startEndPoints(string, second)
%% Function will detect the start and end point of silence (for a given duration)

% INPUT:
% - string:         Predicted label string of 0s and 1s.  1 denotes speech
%                   and 0 denotes non-speech
% - second:         Duration of silence you are looking for (0.5 unit) for
% 2 second it should be 4

% OUTPUT:
% - startIndex:     Column vector including start points of silence
% - endIndex:       Cplumn vector including end points of sulence
% - resultsTable:   Table including start point and end point information for
%                   each silence period

% - Date:           2017-June 9
% - Author:         Viktoria Manukyan

dsig = diff([1 string' 1]);
startIndex = find(dsig < 0);
endIndex = find(dsig > 0)-1;
duration = endIndex-startIndex+1;
stringIndex = (duration >= second);

startIndex = (startIndex(stringIndex))';
endIndex = (endIndex(stringIndex))';
Start_Time = (startIndex-1)*0.5;
End_Time = endIndex*0.5;
for i = 1:length(Start_Time)
    min(i) = floor(Start_Time(i)/60);
    min(i) = floor(Start_Time(i)/60);
    sec(i) = (Start_Time(i) - min(i)*60);
    time = [min' sec'];
end
if startIndex ~= 0
    for i = 1:length(startIndex)
        unit{i} = 'sec';
        rows{i} = ['Silence_',num2str(i)];
        Duration(i) = End_Time(i)- Start_Time(i);
    end

        unit = unit';
        rows = rows';
        Duration = Duration';

    resultsTable = table(startIndex, endIndex, Start_Time, End_Time,time, Duration,unit,'RowNames', rows);
else
    resultsTable = ['No silence period longer than ' , num2str(second*0.5), ' seconds'];
end
end