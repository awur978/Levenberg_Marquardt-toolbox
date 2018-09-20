% Netzaufbau mit Gewichten und Verz�gerungen erstellen und wichtige ngen
% erstellen

function [net]=CreateNN(nn,dIn,dIntern,dOut)

% dIn: Zelle f�r jeden Eingang mit Vektor der Verz�gerungen
if ~exist('dIn', 'var')
    dIn = [0];
end
% dIntern: Vektor mit den Verz�gerungen der Internen Schichten (Au�erSM -> S1)
if ~exist('dIntern', 'var')
    dIntern = [];
end
% dOut: Zelle f�r jeden Ausgang mit Vektor der Verz�gerungen zur Schicht S1
if ~exist('dOut', 'var')
    dOut = [];
end





% NN=[P S1 S2 .. SM] Aufbau des Netzes( P steht f�r Anzahl der
    % Eing�nge, Sm f�r die Anzahl der ronen von Schicht m

    
  
net.nn=nn; %Gesamtaufbau mit Eing�ngen
net.delay.In=dIn;    %Eingansverz�gerung
net.delay.Intern=dIntern;    %Interne Verz�gerungen
net.delay.Out=dOut;    %Ausgangsverz�gerungen

net.M=length(nn)-1;        %Anzahl der Schichten des Neuronalen Netzes
net.layers=nn(2:end);      % Aufbau des NN ohne Eingangsmatrix [S1 S2...SM]
net.dmax=max([net.delay.In,net.delay.Intern,net.delay.Out]); %Maximale Verz�gerung innerhalb des NN


[net]=w_Create(net); %Gesamtgewichtsvektor Net.w_0 erzeugen sowie wichtige Mengen erzeugen

net.N=length(net.w_0); %Anzahl der Gewichte
net.w=net.w_0; %Zu Otimierender Gesamtgewichtsvektor Net.w_k ist zu Beginn der Startvektor Net.w_0

end