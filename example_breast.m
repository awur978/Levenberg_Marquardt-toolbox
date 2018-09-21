%%
%Read Example Data
[x,t]=cancer_dataset;
%Inputs and outputs have to be matrices where columns=datapoints
%and rows=inputs

[I N] = size(x);
[O N]=size(t);
maxt = max(max(t));  

Q = size(x,2); %total number of samples
Q1 = floor(Q * 0.80); %90% for training
Q2 = Q-Q1; %10% for testing
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));
P = x(:, ind1);
Y = t(:, ind1);
Ptest = x(:, ind2);
Ytest = t(:, ind2);
lt = floor(Q*0.8); %training set
lv = ceil(Q*0.2); %validation set
epoch = 1000;

%%
%Create NN

%create feed forward neural network with 9 input, 1 hidden layer with 
%5 neurons each and 2 output
net = CreateNN([9 5 2]); 

%%
%Train with Vanilla LM-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 100
% Set termination condition for Error E_stop to 1e-5
% The Training will stop after 100 iterations or when the Error <=E_stop
netLM = train_LM(P,Y,net,epoch,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
y_LM = NNOut(P,netLM); 
ytest_LM = NNOut(Ptest,netLM); 

%%
%Train with stochasticity-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 200
% Set termination condition for Error E_stop to 1e-5
% The Training will stop after 200 iterations or when the Error <=E_stop
% measure time dt
% netBFGS = train_BFGS(P,Y,net,200,1e-5);
% %Calculate Output of trained net (LM) for training and Test Data
% y_BFGS = NNOut(P,netBFGS); 
% ytest_BFGS = NNOut(Ptest,netBFGS); 


%%
%Plot Results
%plot errors
% err1 = abs(Y-y_LM);
% figure;
% subplot(311), plot(err1(1,:)); grid on;
% title('Training Error');
% subplot(312),plot(err1(2,:));grid on;
% subplot(313),plot(err1(3,:));grid on;
% 
% 
% err = abs(Ytest-ytest_LM);
% figure;
% subplot(311), plot(err(1,:)); grid on;
% title('Testing Error');
% subplot(312),plot(err(2,:));grid on;
% subplot(313),plot(err(3,:));grid on;
% 

figure;
subplot(211);plot(1:lv,Ytest(1,:),'o',1:lv,ytest_LM(1,:),'*');
title('Network''s Performance - Testing');
subplot(212);plot(1:lv,Ytest(2,:),'o',1:lv,ytest_LM(2,:),'*');

figure;
set(gca,'FontSize',16)
plot(netLM.ErrorHistory,'b','LineWidth',2)
title('Training Epoch');
xlabel('Epoch');
ylabel('SSE');
grid on

