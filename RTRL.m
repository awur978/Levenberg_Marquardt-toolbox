% RTRL algorithm
% Calculation of the Jacobian matrix J of the error surface

function [J,E,e,a] = RTRL(net,data) 

P=data.P;   % Input data
Y=data.Y;   % Output data of the system
a=data.a;   % Shift outputs
q0=data.q0; % From total training data


M = net.M;      % Number of layers of the NN
U = net.U;      % Vector with all the initial layers
X = net.X;      % Vector with all input layers
layers=net.layers;    % Structure of the network
dI=net.dI;      % Delay of the inputs
dL=net.dL;      % Delay between layers
L_f=net.L_f;    % Forward connections of the layers
L_b=net.L_b;     % Reverse links
I=net.I;        % Inputs into the layers
CU_LW=net.CU_LW;    % CU_LW {x} Amount of all output layers with a connection to x

%Convert total weight vector to weight matrices / vectors
[IW,LW,b]=w2Wb(net);

%% 1.  Propagate Forward:
[Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0);     %Calculate outputs, total outputs and layer outputs of the NN


%% 2. Calculate cost function:
Y_delta=Y-Y_NN;   % Error Matrix: Error of the NN output with respect to the system output for each data point
e=reshape(Y_delta,[],1);    %Error vector (among each other) [y_delta1_ZP1;y_delta2_ZP1;...;y_delta1_ZPX;y_delta2_ZPX]
E=e'*e; % Cost function: summed square error
% or you can just use
%E=sse(Y_delta);

%% 3. Backpropagation RTRL

Q = size(P,2);     % Number of data points with "old data"
Q0 = Q-q0;  % Number of data points without "old data"

%Predefinitions
dAu_db=cell(M,1);        % Derivative da (u) / db according to bias vectors
dAu_dIW=cell(size(IW)); % Derivative da (u) / dIW after input weight matrices
dAu_dLW=cell(size(LW)); % Derivative da (u) / dLW after connecting weight matrices
S=cell(Q,M);            % Sensitivity matrices
dA_dw=cell(Q,max(U));    % Derivative dA (u) / dw by total weight vector
Cs=cell(max(U),1);  % Set (all m) of the existing sensitivity matrices of layer U: Cs{U}=All m for which S {q, U, m} exists
CsX=cell(max(U),1); %Set (all x) of the existing sensitivity matrices of layer U: CsX {U} = All x for which S {q, U, x} exists
% Cs and CsX are generated during the calculation of the sensitivities

% Initialization
J=zeros(Q0*layers(end),net.N); %Jacobian matrix

for q=1:q0
     for u=U      % For all u € U
        dA_dw{q,u}=zeros(layers(u),net.N);   %Initialization
     end
end

%--------Start RTRL -------------------------------   
    for q=q0+1:Q % All training data from q0 + 1 to Q

        U_=[];  % Quantity Necessary for the calculation of the sensitivities, is generated during the correction
        for u=U     % For all u € U
            Cs{u}=[];       %Initialization
            CsX{u}=[];      %Initialization
            dA_dw{q,u}=0;   %Initialization
        end

%---------------Calculate Sensitivity Matrices----------------------------

        for m=M:-1:1  % m decrement in backpropagation order

            for u=U_    %all u € U_
                S{q,u,m}=0; %% Sensitivity matrix from layer u to layer m
                for l=L_b{m} % All layers with direct reverse connection Lb (m) to the layer m
                    S{q,u,m}=S{q,u,m}+(S{q,u,l}*LW{l,m,1})*diag((1-((tanh(n{q,m})).^2)),0); %Calculate Sensitivity Matrix Recursively
                end
                if all(m~=Cs{u})  % If m is not yet in Cs {u}
                    Cs{u}=[Cs{u},m];    % Add m to the set Cs (u)
                    if any(m==X)  % If m € X
                        CsX{u}=[CsX{u},m];  % Add m to the set Csx (u)
                    end
                end
            end
            if any(m==U)  % If m € U
                if m==M % If m = M (output layer M has no transfer function: a {M} = n {M})
                    S{q,m,m}=diag(ones(layers(M),1),0);  %Calculate sensitivity matrix S (M, M)
                else
                    S{q,m,m}=diag((1-((tanh(n{q,m})).^2)),0); %Calculate sensitivity matrix S (m, m)
                end
                U_=[U_,m];  % Add m to the set U '
                Cs{m}=[Cs{m},m];  % Add m to the set Cs (m)
                if any(m==X) %If m € X
                    CsX{m}=[CsX{m},m];  %Add m to the set Csx (m)
                end
            end
        end
        
%-------------- Derivatives Calculate----------------------------------------            
        for u=sort(U)     %% All u € U increments in simulation order
          
                    %------------Static derivation calculation ----------------------- 
           dAe_dw=[]; %Explicit derivative outputs layer u after all weights
           for m=1:M  %All layers m            
                %------------------
                %Input weights
                if m==1
                    for i=I{m}  %All inputs i couple in layer m
                        for d=dI{m,i}   % all delays i-> m
                            if (sum(size(S{q,u,m}))==0)||(d>=q) % If there is no sensitivity OR d> = q:
                                dAu_dIW{m,i,d+1}=kron(P(:,q)',zeros(layers(u),layers(m)));   %Derivative equals zero
                            else
                                dAu_dIW{m,i,d+1}=kron(P(:,q-d)',S{q,u,m});   %Derivative output u after IW {m, i, d + 1}
                            end
                            dAe_dw=[dAe_dw,dAu_dIW{m,i,d+1}]; % Append % to total derivative vector da (u) / dw
                        end
                    end
                end
                %---------------------
                %Connection weights
                for l=L_f{m}  %All layers that have a direct forward connection to layer m
                    for d=dL{m,l}    % all delays l-> m
                        if (sum(size(S{q,u,m}))==0)||(d>=q) %If there is no sensitivity OR d> = q:
                            dAu_dLW{m,l,d+1}=kron(a{q,l}',zeros(layers(u),layers(m)));   %Derivative equals zero
                        else
                            dAu_dLW{m,l,d+1}=kron(a{q-d,l}',S{q,u,m});   %Derivative output u to LW {m, l, d + 1}
                        end
                        dAe_dw=[dAe_dw,dAu_dLW{m,l,d+1}]; % Append to total derivative vector da (u) / dw
                    end
                end
                %--------------
                %Bias weights
                if (sum(size(S{q,u,m}))==0) % If there is no sensitivity
                    dAu_db{m}=zeros(layers(u),layers(m));%Derivative output u to b {m} = NULL
                else
                    dAu_db{m}=S{q,u,m};     % Derivative output u to b {m}
                end
                dAe_dw=[dAe_dw,dAu_db{m}]; % Append % to total derivative vector da (u) / dw
           end %end m

         
    %-----------dynamic derivation calculation------------------------------

            dAd_dw=0; %Total over all x
            for x=CsX{u} % All x in CsX (u)
                sum_u_=0;  %Sum over all u_
                for u_=CU_LW{x} %all u_ in CU_LW {x}
                    sum_d=0; %Sum over all d
                    for d=dL{x,u_} %All delays layer_ on layer x
                        if ((q-d)>0)&&(d>0) %delay can not be greater than current date, only delays> 0
                            sum_d=sum_d+LW{x,u_,d+1}*dA_dw{q-d,u_}; %Sum over all d
                        end
                    end
                    sum_u_=sum_u_+sum_d; %Sum over all u_
                end
                if abs(sum(sum(sum_u_)))>0 %If sum_u is valid
                    dAd_dw=dAd_dw+S{q,u,x}*sum_u_; %Sum up dynamic derivation calculation
                end
            end

      %-------- Static + Dynamic Share-----------------------  

            dA_dw{q,u}=dAe_dw+dAd_dw;   %Total derivation calculation Output u to total weight vector w

        end %end u
    
    %------ occupy Jacobian matrix-----------------------------
        J(((q-q0)-1)*layers(end)+1:(q-q0)*layers(end),:)=-dA_dw{q,M};         %Jacobian matrix
             
    end  %end q 
end
