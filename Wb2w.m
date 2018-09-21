% Input and connection matrices and bias vectors for in Net
% summarize  given in total weight vector net.w_0

function w=Wb2w(net,IW,LW,b)

dL=net.dL; % Delays between layers
dI=net.dI; % Delays of the inputs
I=net.I;       % Input matrices (here only one input matrix coupled in layer 1)
L_f=net.L_f;  % Forward links
M=net.M; % Number of hits of the NN

w=[];  % Total weight vector

for m=1:M  % All layers m
    
    % Input weights
    if m==1 
        for i=I{m}  % All input matrices i that couple in layer m
            for d=dI{m,i}   % all delays i-> m
                w=[w;IW{m,i,d+1}(:)]; % Add input weight matrix to total weight vector [Matix (:) = vec (Matric)]
            end
        end
    end

    % Connection weights
    for l=L_f{m} % All layers that have a direct forward link to the layer
        for d=dL{m,l}    % all delays l-> m
            w=[w;LW{m,l,d+1}(:)]; %Add connection weight matrix to total weight vector [Matix (:) = vec (Matric)]
        end
    end
    
    %Bias weights
    w=[w;b{m}];     %Add bias vector from layer m to the total weight vector
end