% Calculate Error for NN based on data
% 
% Args:
%     net:	neural network
%     data: 	Training Data
% Returns:
%     E:		Mean squared Error of the Neural Network compared to Training data

function [E] = calc_error(net,data) 


P=data.P;   %Eingangsdaten
Y=data.Y;   %Ausgangsdaten des Systems
a=data.a;   %Schichtausg�nge
q0=data.q0; %Ab q0.tem Trainingsdatum Ausg�nge berechnen


M = net.M;      %Anzahl der Schichten des NN
layers=net.layers;    %Aufbau des Netzes
dI=net.dI;      %Verz�gerung der Eing�nge
dL=net.dL;      %Verz�gerung zwischen den Schichten
L_f=net.L_f;    %Vorw�rtsverbindungen der Schichten

I=net.I;        %Eing�nge in die Schichten


% Gesamtgewichtvektor in Gewichtmatrizen/ Vektoren umwandeln
[IW,LW,b]=w2Wb(net);

%% 1. Vorw�rtspropagieren:
[Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0);     %Ausg�nge, Summenausg�nge und Schichtausg�nge des NN berechnen


%% 2. Kostenfunktion berechnen:
Y_delta=Y-Y_NN;   %Fehlermatrix: Fehler des NN-Ausgangs bez�glich des Systemausgangs f�r jeden Datenpunkt
e=reshape(Y_delta,[],1);    %Fehlervektor (untereinander) [y_delta1_ZP1;y_delta2_ZP1;...;y_delta1_ZPX;y_delta2_ZPX]
E=e'*e; %Kostenfunktion: Summierter Quadratischer Fehler