% Input and connection matrices and bias vectors for in Net
% preset network structure and in total weight vector Net.w_0
% Summarized
% In addition, several important quantities are defined
function [net]=w_Create(net)

% Randomize random numbers
ms_time=str2num(datestr(now,'FFF'));
RStr = RandStream('mcg16807','Seed',ms_time);
RandStream.setGlobalStream(RStr);


M=net.M;    %Number of layers of the NN
layers=net.layers; % Structure of the NN
inputs=net.nn(1); %Number of input
delay=net.delay; %Delays

%Definitions
X=[];   % Amount of input layers (input weights or delays> 0 for connection weights)
U=[];   % Amount of output layers (output of the chirp is included in the calculation of the KF or go via delay> 0 in an input layer)
I=cell(M,1); %Amount of inputs coupling into layer m



%---------------------------
%  Inputs only couple into layer 1
dI{1,1}=delay.In;   %Input delays from layer 1 to input 1
for d=dI{1,1}   %all input delays d
    IW{1,1,d+1}= (-0.5 + 1.*rand(layers(1),inputs));    %Input weight matrix P-> S1 IW {d} stands for the delay [d-1], since Matlab does not take zero as an index
end 
X=[1];  %first layer is input layer since the system inputs couple here
I{1}=1; % Amount of inputs feeding into layer 1 (one input only)

%---------------------------------------
%Create connection weight matrices
for m=1:M %All layers m
    L_b{m}=[]; % Amount of layers having a direct reverse connection to layer m
    L_f{m}=[]; % Amount of layers having a direct forward connection to layer m
    
    %Forward links
    if m>1  %
        l=m-1;
        dL{m,l}=0;    % no delay in the forward links
        LW{m,l,1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Weight matrix Sm -> Sm + 1 forward link only to the next layer and without delay. Random values ??between -0.5 and 0.5
        L_b{l}=m; % Layer m has a direct reverse connection to layer l
        L_f{m}=[L_f{m},l]; %Layer I has a direct forward connection to layer m
    end
    
    %Reverse links
    for l=m:M % There are possible back links to the layer m of all layers> = m
        
        if (m==1)&&(l==M)   %Special case: Delay of output to layer 1
            dL{m,l}=delay.Out; %Delays of the output layer to the input layer 1
        else
            dL{m,l}=delay.Intern; %All other Schcihten have to yourself and to all previous layers from the delays delay.Intern
        end
        
        for d=dL{m,l} %All delays from layer 1 to layer m
            LW{m,l,d+1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Weight matrix Sl -> Sm for create delay d. Random values ??between -0.5 and 0.5
            if (sum(l==L_f{m})==0) % If l is not already present in L_f {m}
                L_f{m}=[L_f{m},l];  %Add l to the set L_f {m}
            end
            if (l>=m)&&(d>0) % If LW {m, l, d + 1} a Delayed Feedback
                if (sum(m==X)==0) %And if m does not exist in X yet
                    X=[X,m];    %Add % m to the amount of input layers
                end
                if (sum(l==U)==0) % And if l does not exist in U yet
                    U=[U,l];    % Add % l to the amount of starting layers
                end
            end
        end
    end
    
    b{m}=(-0.5 + 1.*rand(layers(m),1)); % Create % bias vector layer m. Random values ??between -0.5 and 0.5
end
            
if (sum(M==U)==0) %If M is not yet present in U
    U=[U,M];    % last layer is output layer, since system output
end       

for  u=U %For all starting layers
    CX_LW{u}=[]; %Amount of all input layers that receive a signal from u
    for x=X %for all input layers
        if (size(intersect(u,L_f{x}))>0)&(sum(x==CX_LW{u})==0)&(any(dL{x,u}>0)) %If u in L_f {x} AND x not yet in CX_LW {u} AND the compound layer u -> layer x has a delay> 0
            CX_LW{u}=[CX_LW{u},x];   % Add % u to the set CU_LW {x}
        end
    end
end

for x=1:M %For all layers
    CU_LW{x}=[]; %Amount of all starting layers connected to x
    for u=U %For all starting layers
        if any(dL{x,u}>0)   %If the compound layer u -> layer x has a delay> 0 
            CU_LW{x}=[CU_LW{x},u];   %Add % c to the set CX_LW {u}
        end
    end
end

net.U=U; % Amount of all starting layers
net.X=X; % Amount of all input layers
net.dL=dL; % Delays between layers
net.dI=dI; % Delays of the inputs
net.L_b=L_b; %Reverse links
net.L_f=L_f;  % Forward links
net.I=I;      % Input matrices (here only one input matrix coupled in layer 1)
net.CU_LW=CU_LW;  %CX_LW {u} =% Amount of all input layers that receive a signal from u    
net.CX_LW=CX_LW; % CU_LW {x} = set of all output layers connected to x

net.w_0=Wb2w(net,IW,LW,b);  % Create the total weight vector from input and connection matrices and bias vectors
                    
    

