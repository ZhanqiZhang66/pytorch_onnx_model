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
Z = dlarray(zeros(224,224,1,1), 'SSCB');
hiddenout = ConvNet.predict(Z);
hiddenout_r = dlarray(hiddenout.reshape(1,9,224,224).permute([1,2,5,3,6,4]),"SSCB"); % Note to flip your hiddenoutput

out = DeconvNet.predict(hiddenout_r);

imgs = extractdata(out(:,:,:,:));
