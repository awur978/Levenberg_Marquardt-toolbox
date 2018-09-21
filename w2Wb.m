% Conversion of the total weight vector w into the input weight matrices IW,
% the connection weight matrices LW and the bias vectors b

function [IW,LW,b]=w2Wb(net)

I=net.I;                % Inputs into the layers
dI=net.dI;              % Delay of the inputs
L_f=net.L_f;           % Forward connections of the layers
dL=net.dL;              % Delay between layers
inputs=net.nn(1); % Number of inputs
layers=net.layers;            % Structure of the network
w_temp=net.w;         % temporary total weight vector
M=net.M;                % Number of layers of the NN

% Predefinitions
b=cell(M,1);                % Bias vectors   
IW=cell(1,1,max(dI{1,1}));  % Input weight matrices
LW=cell(M,M,net.dmax);     % Connection weight matrices

for m=1:M  % All layers m
      
    %Input weights
    if m==1
        for i=I{m}  % All inputs i in layer m
            for d=dI{m,i}   % all delays i-> m
                w_i=inputs*layers(m);  % Number of weights of the input matrix IW {m, i, d + 1} (matrix of input i to layer m for delay d)
                vec=w_temp(1:w_i);  % Read elements from total weight vector
                w_temp=w_temp(w_i+1:end);   %Remove elements from temporary total weight vector
                IW{m,i,d+1}=reshape(vec,layers(m),[]);%Migrate Read Elements from Vector vec to Matrix
            end
        end
    end

    %Connection weights
    for l=L_f{m}  % All inputs i
        for d=dL{m,l}   %  all delays i-> m
            w_i=layers(l)*layers(m); % Number of weights of the input matrix IW {m, i, d + 1} (matrix of input i to layer m for delay d)
            vec=w_temp(1:w_i);  % Read elements from total weight vector
            w_temp=w_temp(w_i+1:end);   %Remove elements from temporary total weight vector
            LW{m,l,d+1}= reshape(vec,layers(m),[]);%Migrate Read Elements from Vector vec to Matrix
        end
    end
    
    %Bias weights
    w_i=layers(m); % Number of weights of the bias vector of the layer m
    b{m}=w_temp(1:w_i); % ead% elements from total weight vector
    w_temp=w_temp(w_i+1:end);  %Remove elements from temporary total weight vector
end
