% Eingangs- und Verbindungsmatrizen sowie Biasvektoren f�r in Net
% vorgegebene Netzstruktur ertsellen und in Gesamtgewichtsvektor Net.w_0
% zusammenfassen
% Zus�tzlich werden verschiedenen wichtige Mengen definiert
function [net]=w_Create(net)

%Zufallszahlen zuf��lig machen
ms_time=str2num(datestr(now,'FFF'));
RStr = RandStream('mcg16807','Seed',ms_time);
RandStream.setGlobalStream(RStr);


M=net.M;    %Anzahl der Schichten des NN
layers=net.layers; % Aufbau des NN
inputs=net.nn(1); %Anzahl 
delay=net.delay; %Verz�gerungen

%Definitionen
X=[];   % Menge der Eingangsschichten( Eingangsgewichte oder Verz�gerungen >0 bei Verbindungsgewichten)
U=[];   % Menge der Ausgangsschichten (Ausgang der Schihct geht in Berechnung der KF ein oder geh �ber Verz�gerung > 0 in eine Eingangsschicht)
I=cell(M,1); %Menge der Eing�nge, die in Schicht m einkoppeln



%---------------------------
% Eing�nge koppeln nur in Schicht 1 ein
dI{1,1}=delay.In;   %Eingangsverz�gerungen vonzu Schicht 1 von Eingang 1
for d=dI{1,1}   %all Eingangsverz�gerungen d
    IW{1,1,d+1}= (-0.5 + 1.*rand(layers(1),inputs));    %Eingangs-Gewichtmatrix P->S1    IW{d} steht dabei f�r die Verz�gerung [d-1], da Matlab keine Null als Index nimmt
end 
X=[1];  %erste Schicht ist Eingangsschicht, da die Systemeinga�nge hier einkoppeln
I{1}=1; % Menge der Eing�nge, die in Schicht 1 einkoppeln (Nur ein Eingang)

%---------------------------------------
%Verbindungdgewichtsmatrizen erstellen
for m=1:M %Alle Schichten m
    L_b{m}=[]; % Menge der Schichten , die eine direkte R�ckw�rtsverbindung zu Schicht m besitzen
    L_f{m}=[]; %Menge der Schichten, die eine direkte Vorw�rtsverbindung zur Schicht m besitzen
    
    %Vorw�rtsverbindungen
    if m>1  %
        l=m-1;
        dL{m,l}=0;    %keine Verz�gerun in den Vorw�rtsverbindungen
        LW{m,l,1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Gewichtmatrix Sm -> Sm+1 Vorw�rtsverbindung nur zur n�chsten Schicht und ohne Verz�gerung. Zuf�llige Werte zwischen -0,5 und 0,5
        L_b{l}=m; % Schicht m besitzt eine direkte R�ckw�rtsverbindung zur Schicht l
        L_f{m}=[L_f{m},l]; %Schicht l besitzt eine direkte Vorw�rtsverbindung zur Schicht m
    end
    
    %R�ckw�rtsverbindungen
    for l=m:M % Es gibt m�gliche R�ckw�rtserbindungen zur Schicht m von allen Schichten >= m
        
        if (m==1)&&(l==M)   %Sonderfall: Verz�gerung von Ausgang zur Schicht 1
            dL{m,l}=delay.Out; %Verz�gerungen der Ausgangsschicht zur Eingangsschicht 1
        else
            dL{m,l}=delay.Intern; %Alle anderen Schcihten habe zu sich selbst und zu allen vorherigen Schichten die Verz�gerungen aus delay.Intern
        end
        
        for d=dL{m,l} %Alle Verz�gerungen von Schicht l zur Schicht m
            LW{m,l,d+1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Gewichtmatrix Sl -> Sm f�r f�r Verz�gerung d erstellen. Zuf�llige Werte zw -0,5 und 0,5
            if (sum(l==L_f{m})==0) % Falls l noch nicht in L_f{m} vorhanden
                L_f{m}=[L_f{m},l];  %l zur Menge L_f{m} hinzuf�gen
            end
            if (l>=m)&&(d>0) % Falls LW{m,l,d+1} eine Verz�gerte R�ckkopplung
                if (sum(m==X)==0) %Und falls m noch nicht in X vorhanden
                    X=[X,m];    % m zur Menge der Eingangsschichten hinzuf�gen
                end
                if (sum(l==U)==0) % Und falls l noch nicht in U vorhanden 
                    U=[U,l];    % l zur Menge der Ausgangsschichten hinzuf�gen
                end
            end
        end
    end
    
    b{m}=(-0.5 + 1.*rand(layers(m),1)); % Biasvektor Schicht m erzeugen. Zuf�llige Werte zw. -0,5 und 0,5
end
            
if (sum(M==U)==0) %Falls M noch nicht in U vorhanden
    U=[U,M];    % letzte Schicht ist Ausgangsschicht, da Systemausgang
end       

for  u=U %F�r alle Ausgangsschichten
    CX_LW{u}=[]; %Menge aller Eingangsschichten, die ein Signal von u bekommen
    for x=X %f�r alle Eingangsschichten
        if (size(intersect(u,L_f{x}))>0)&(sum(x==CX_LW{u})==0)&(any(dL{x,u}>0)) %Falls u in L_f{x} UND x noch nicht in CX_LW{u} UND die Verbindung Schicht u -> Schicht x eine Verz�gerung >0 besitzt
            CX_LW{u}=[CX_LW{u},x];   %c zur Menge CX_LW{u} hinzuf�gen
        end
    end
end

for x=1:M %F�r alle Schichten
    CU_LW{x}=[]; %Menge aller Ausgangsschichten mit einer Verbindung zu x
    for u=U %F�r alle Ausgangsschichten
        if any(dL{x,u}>0)   %Falls die Verbindung Schicht u -> Schicht x eine VErz�gerung >0 besitzt 
            CU_LW{x}=[CU_LW{x},u];   %u zur Menge CU_LW{x} hinzuf�gen
        end
    end
end

net.U=U; % Menge aller Ausgangsschichten
net.X=X; % Menge aller Eingangsschichten
net.dL=dL; % Verz�gerungen zwischen den Schichten
net.dI=dI; % Verz�gerungen der Eing�nge
net.L_b=L_b; %R�ckw�rtsverbindungen
net.L_f=L_f;  % Vorw�rtsverbindungen
net.I=I;      % Eingangsmatrizen (hier nur eine Eingangsmatrix der in Schicht 1 einkoppelt)
net.CU_LW=CU_LW;  %CX_LW{u} = %Menge aller Eingangsschichten, die ein Signal von u bekommen  
net.CX_LW=CX_LW; %CU_LW{x} = Menge aller Ausgangsschichten mit einer Verbindung zu x

net.w_0=Wb2w(net,IW,LW,b);  %Gesamtgewichtsvektor aus Eingangs- und Verbindungsmatrizen sowie Biasvektoren erstellen
                    
    

