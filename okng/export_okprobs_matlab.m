% export_okprobs_matlab.m
% -------------------------------------------------------------------------
% Template: produce the OK-probability tables that the temporal pipeline's
% inference-time OK/NG gate (configs B / D) consumes, using YOUR OWN OK/NG
% model saved as a MATLAB .mat (e.g. a DAGNetwork / dlnetwork classifier).
%
% Output: one file per fold,  okprobs_fold{1..5}.csv  with columns:
%     path      - frame filename (CaseXXYY_HHMMSS.jpg)
%     ok_prob   - probability that the frame is OK (usable), in [0,1]
% Place these CSVs in the directory you pass as AMVIDEO_OKPROB_DIR.
%
% This is methodology glue: the temporal code is model-agnostic and only
% needs the (path, ok_prob) table. Edit the marked sections for your model.
% -------------------------------------------------------------------------

% --- 1. Load your OK/NG model -------------------------------------------
S = load('your_okng_model.mat');          % <-- EDIT: path to your .mat
net = S.net;                              % <-- EDIT: variable holding the network
okClassName = "OK";                       % <-- EDIT: name of the OK class label
inputSize = net.Layers(1).InputSize(1:2); % e.g. [299 299] for Inception-ResNet-v2

% --- 2. For each fold, list its test-case frames ------------------------
% Point this at the folders (or a file list) of frames for each fold's test
% cases. Each frame filename must be CaseXXYY_HHMMSS.jpg.
foldFrameDirs = { ...
    'frames/fold1', 'frames/fold2', 'frames/fold3', ...
    'frames/fold4', 'frames/fold5'};       % <-- EDIT

outDir = 'okprobs';                        % <-- EDIT: = AMVIDEO_OKPROB_DIR
if ~exist(outDir, 'dir'); mkdir(outDir); end

for f = 1:numel(foldFrameDirs)
    files = dir(fullfile(foldFrameDirs{f}, '**', 'Case*.jpg'));
    n = numel(files);
    paths   = strings(n, 1);
    okprobs = zeros(n, 1);

    for i = 1:n
        fpath = fullfile(files(i).folder, files(i).name);
        img = imresize(imread(fpath), inputSize);

        % --- 3. Predict OK probability for this frame -------------------
        scores = predict(net, img);                 % 1 x numClasses
        okIdx  = (string(net.Layers(end).Classes) == okClassName);
        okprobs(i) = scores(okIdx);

        paths(i) = string(files(i).name);
    end

    T = table(paths, okprobs, 'VariableNames', {'path', 'ok_prob'});
    writetable(T, fullfile(outDir, sprintf('okprobs_fold%d.csv', f)));
    fprintf('wrote okprobs_fold%d.csv (%d frames)\n', f, n);
end
