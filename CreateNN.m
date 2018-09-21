% Create network structure with weights and delays and attach important ones Create %

function [net]=CreateNN(nn,dIn,dIntern,dOut)

% dIn: cell for each input with vector of delays
if ~exist('dIn', 'var')
    dIn = [0];
end
% dIntern: Vector with the delays of the internal layers (except SM -> S1)
if ~exist('dIntern', 'var')
    dIntern = [];
end
% dOut: cell for each output with delay line vector S1
if ~exist('dOut', 'var')
    dOut = [];
end





%  NN = [P S1 S2 .. SM] Structure of the network (P stands for number of
    % Inputs, Sm for the number of rons of layer m


    
  
net.nn=nn; %Overall construction with inputs
net.delay.In=dIn;    % Input delay
net.delay.Intern=dIntern;    %Internal delays
net.delay.Out=dOut;    %Exit delays

net.M=length(nn)-1;        % Number of layers of the neural network
net.layers=nn(2:end);      % Structure of NN without input matrix [S1 S2 ... SM]
net.dmax=max([net.delay.In,net.delay.Intern,net.delay.Out]); %Maximum delay within the NN


[net]=w_Create(net); %Generate total weight vector Net.w_0 and generate significant quantities

net.N=length(net.w_0); %Number of weights
net.w=net.w_0; %Net.w_k to be initialized is the start vector Net.w_0 at the beginning

end