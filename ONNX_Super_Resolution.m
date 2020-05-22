clc
modelfile = 'super_resolution.onnx';
layers = importONNXLayers(modelfile,'OutputLayerType','classification','ImportWeights',true) ; 
%%
nets = importONNXNetwork(modelfile, 'OutputLayerType', 'classification');
%%
analyzeNetwork(layers)
%%
net = assembleNetwork(layers);
%%
%previous the 8-th node
ConvNet=dlnetwork(layerGraph(layers.Layers(1:8)));

hiddeninputlayer = imageInputLayer([224*3 224*3 1],'Name','hiddeninput','Normalization','none');
DeconvNet = dlnetwork(layerGraph([hiddeninputlayer; layers.Layers(12:13)]));
%%
img = imread("E:\Monkey_Data\Generator_DB_Windows\nets\deepsim\Cat.jpg");
img = imresize(img,[224,224]);
%%
% Z = dlarray(zeros(224,224,1,1), 'SSCB');
Z = dlarray(single(reshape(img,[224,224,1,3]))/255, 'SSCB');%mean(img,3)
hiddenout = ConvNet.predict(Z);
ps_out = pixelshuffle(hiddenout, 3);
ps_out = squeeze(ps_out);
% hiddenout_r = dlarray(hiddenout.reshape(1,9,224,224).permute([1,2,5,3,6,4]),"SSCB"); % Note to flip your hiddenoutput
%%
figure;imshow(ps_out.extractdata)
%%
figure;imshow(imresize(img,[224*3,224*3]))
%%
out = DeconvNet.predict(hiddenout_r);

imgs = extractdata(out(:,:,:,:));


function out = pixelshuffle(input, upscale)
if nargin == 1
    upscale = 2;
end
H = size(input,1); W = size(input,2);
Ch = size(input,3) / upscale.^2;
B = size(input,4);
input_fact = reshape(input,[H,W,upscale,upscale,Ch,B]);
out = reshape(permute(input_fact, [4,1,3,2,5,6]), [H * upscale,W * upscale,Ch,B]);
end