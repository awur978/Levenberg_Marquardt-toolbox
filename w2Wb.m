% Umwandeln des Gesamtgewichtsvektors w in die Eingangsgewichtsmatrizen IW,
% die Verbindungsgewichtsmatrizen LW und die Biasvektoren b


function [IW,LW,b]=w2Wb(net)

I=net.I;                %Eing�nge in die Schichten
dI=net.dI;              %Verz�gerung der Eing�nge
L_f=net.L_f;            %Vorw�rtsverbindungen der Schichten
dL=net.dL;              %Verz�gerung zwischen den Schichten
inputs=net.nn(1); %Anzahl der Eing�nge
layers=net.layers;            %Aufbau des Netzes
w_temp=net.w;         %tempor�rer Gesamtgewichtsvektor
M=net.M;                %Anzahl der Schichten des NN

%Vordefinitionen
b=cell(M,1);                %Biasvektoren    
IW=cell(1,1,max(dI{1,1}));  %Eingangsgewichtsmatrizen
LW=cell(M,M,net.dmax);      %Verbindungsgewichtsmatrizen

for m=1:M  %Alle Schichten m
      
    %Eingangsgewichte
    if m==1
        for i=I{m}  %Alle Eing�nge i in Schicht m
            for d=dI{m,i}   % alle Verz�gerungen i->m
                w_i=inputs*layers(m);  % Anzahl von Gewichten der Eingangsmatrix IW{m,i,d+1} (Matrix von Eingang i zu Schicht m f�r Verz�gerung d)
                vec=w_temp(1:w_i);  % Elemente aus Gesamtgewichtsvektor auslesen
                w_temp=w_temp(w_i+1:end);   %Elemente aus tempor�ren Gesamtgewichtsvektor entfernen
                IW{m,i,d+1}=reshape(vec,layers(m),[]);%Ausgelesenen Elemente von Vektor vec in Matrix umwamndeln
            end
        end
    end

    %Verbindungsgewichte
    for l=L_f{m}  %Alle Eing�nge i
        for d=dL{m,l}   % alle Verz�gerungen i->m
            w_i=layers(l)*layers(m);  % Anzahl von Gewichten der Verbindungssmatrix LW{m,l,d+1} (Matrix von Schicht l zu Schicht m f�r Verz�gerung d)
            vec=w_temp(1:w_i);  % Elemente aus Gesamtgewichtsvektor auslesen
            w_temp=w_temp(w_i+1:end);   %Elemente aus tempor�ren Gesamtgewichtsvektor entfernen
            LW{m,l,d+1}= reshape(vec,layers(m),[]);%Ausgelesenen Elemente von Vektor vec in Matrix umwamndeln
        end
    end
    
    %Biasgewichte
    w_i=layers(m); % Anzahl von Gewichten des Biasvektors der Schicht m
    b{m}=w_temp(1:w_i); % Elemente aus Gesamtgewichtsvektor auslesen
    w_temp=w_temp(w_i+1:end);  %Elemente aus tempor�ren Gesamtgewichtsvektor entfernen
end
