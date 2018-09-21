% Levenberg Marquardt (LM) with Jacobian matrix calculation by RTRL file

function net=train_LM(P,Y,net,k_max,E_stop)

dampconst   =     10;   %constant to adapt damping factor of LM
dampfac    =     3;    %damping factor of LM

[data,net] = prepare_data(P,Y,net);

[J,E,e]=RTRL(net,data);  %Calculate Jacobian matrix, cost function and error vector of the start vector

k=1; %First iteration step
Ek(k)=E; %Course of the cost function
disp(['Iteration: ', num2str(k),'   Error: ', num2str(E),'   Damping Factor:', num2str(dampfac)])

while 1

    JJ=J'*J; %J (transp) * J
    w=net.w;    % Save current weight vector
    
    while 1 %Until optimization step successful:
        
%         G=inv(JJ+SkalFakt.*eye(size(JJ,1))); %Calculate Scaled Inverse Hessematrix
         G=(JJ+dampfac.*eye(size(JJ,1)))\eye(size(JJ+dampfac.*eye(size(JJ,1))));
        
        g=J'*e;  %Gradient
        if isnan(G(1,1))
            w_delta=-1/1e10.*g;
        else
            w_delta=-G*g;  %Determine change in weight: w_delta = -G * g
        end
        net.w=w+w_delta; %Adjust weights
    
        [E2] = calc_error(net,data); %Calculate cost function on new weight vector

        %-----------Optimization step successful------------------------
        if E2<E     
            dampfac=dampfac/dampconst;    %Adjust scaling factor
            break;                          %Next
            
        %---------%Optimization step NOT successful--------------------   
        elseif E2>=E    
            dampfac=dampfac*dampconst;     %Adjust % scaling factor        
        end          
    end
    
    [J,E,e,a]=RTRL(net,data);  %Jacobian matrix, calculate cost function and error vector to calculate new weight vector
    
    
    k=k+1;  %Current iteration step
    Ek(k)=E; %Course of the cost function
    disp(['Iteration: ', num2str(k),'   Error: ', num2str(E),'   Damping Factor:', num2str(dampfac)])

    if (k>=k_max) || (E<=E_stop) % abort if one of the abort criteria is met
           break
    end    
    
end

%Output
net.ErrorHistory=Ek;

