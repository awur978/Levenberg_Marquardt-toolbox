%Liniensuche mit Lagrange Interpolation
%Finde eta_k f�r die Lagrange interpolation L vom grad r , dass E gleich Minimal


function [eta_k] = ALIS(net,data,d_k,E_k,k,r)
w_k=net.w; %Aktueller Gesamtgewichtsvektor

persistent eta_max; %Obergrenze G�ltigkeitsbereich: Wert zwischen Funktionsaufrufen Speichern
if k==1
    eta_max = 0.5;  %F�r ersten funktionsaufruf setzen 
end
eta_valid =1/4 *eta_max;    %Untergrenze G�ltigkeitsbereich

E=zeros(1,r+1); %Definition E. Es werden r+1 Kostenfunktionswerte E ben�tigt
E(1)=E_k;       % E f�r aktuellen Gewichtvektor bereits bekannt. Ek Kostenfunktion f�r w=w_k)
eta_j=[0:r]./r.*eta_max;  %St�tzstellen berechnen

for J=2:r+1;    %Kostenfunktion f�r die St�tzstellen f�r Lagrange interpolation berechnen
    w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor+Suchrichtung*aktuelleSt�tzstelle_j
    net.w=w;
    E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen St�tzstelle
    
end

%Minimumsuche min E(eta_k) f�r E(w_k+eta_k*d_k)
options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen f�r fminbnd
eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;



%% Intervallgrenzen anpassen, bis gefundenes Minimum im G�ltigkeitsbereich
while 1
    %-----------------------------------------
    if eta_k>=eta_max  %Intervallvergr��erung

        eta_max=2*eta_max;  %Intervallgrenze verdoppeln
        eta_valid=2*eta_valid;  %G�ltigkeitsbereich verdoppeln

        eta_j(1:r/2+1)=([0:r/2]+2)./(2*r).*eta_max; %alte, bereits bekannte Werte �bernehmen
        E(1:r/2+1)=E([1:r/2+1]+2);    %alte, bereits bekannte Werte �bernehmen (r=4: E1=E3; E2=E4 ;E3=E5)

        eta_j(r/2+2:r+1)=([r/2+1:r])./r.*eta_max;   %neue Werte berechnen
        j_min_E=r/2+2;  %Grenzen f�r ALIS_LI bestimmen, dass bereits bekannte Werte nicht nochmal berechnet werden
        j_max_E=r+1;   
        for J=j_min_E:j_max_E;    %Kostenfunktion E nur berechnen, wo noch nicht bekannt
            w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor*Suchrichtung*aktuelleSt�tzstelle_j
            net.w_k=w;
            E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen St�tzstelle
        end

        options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen f�r fminbnd
        eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Neues Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;
        continue;
    end
    
    %-----------------------------------------------
    if eta_k < eta_valid  %Intervallverkleinerung

        eta_max=1/4*eta_max; %Intervallgrenze vierteln
        eta_valid=1/4*eta_valid; %G�ltigkeitsbereich vierteln

       eta_j(r/2+2:r+1)=([r/2+1:r]-2).*eta_max; %alte Werte �bernehmen (Index 1 bleibt gleich!,E1)
        E(r/2+2:r+1)=E((r/2+2:r+1)-2); %alte, bereits bekannte Werte �bernehmen (r=4: E4=E2; E5=E3; E1 bleibt E1)

        eta_j(2:r/2+1)=(1:r/2)./r.*eta_max; %neue Werte berechnen
        j_min_E=2;  %Grenzen f�r ALIS_LI bestimmen, dass bereits bekannte Werte nicht nochmal berechnet werden
        j_max_E=r/2+1;    
        for J=j_min_E:j_max_E;    %St�tzstellen f�r Lagrange interpolation
            w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor*Suchrichtung*aktuelleSt�tzstelle_j
            net.w_k=w;
            E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen St�tzstelle
        end

        options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen f�r fminbnd
        eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Neues Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;
        continue;
    end
    
    % eta_k liegt im G�ltigkeitsbereich
    if (eta_k<=eta_max)&&(eta_k>=eta_valid)
        break; % Funktion beenden und eta_k ausgeben
    end
end


%% Hilfsfunktion ALIS_LI mit nur einem Parameter eta um Minimum  im angegebenen Bereich einfach berechnen zu k�nnen (nested function!)

function L=ALIS_LI(eta)  %Lagrangeinterpolation mit r+1 St�tzstellen in Abh�ngigkeit von eta
    
    L_j=zeros(1,r+1); %Definition: "Vorterm" f�r Lagrange Interpolation jeder St�tzstelle
    
    for j=1:r+1;    %F�r alle St�tzstellen der Lagrange interpolation
    L_h=zeros(1,r+1); %Produkt-Term der LAgrangeinterpolation f�r jede St�tzstelle
        for h=1:r+1
            if h==j 
                 L_h(j)=1;
                 continue;
            end
            L_h(h)= (eta-eta_j(h))/(eta_j(j)-eta_j(h)); % Wird Ben�tigt zur Berechnung vom "Vorterm" f�r Lagrange Interpolation.
        end 
        L_j(j) = prod(L_h);     %"Vorterm" f�r Lagrange Interpolation an St�tzstelle j   Lj(j)=Produkt(j=1:r+1)[L_h(j)]
    end
    L=sum(L_j.*E); %Lagrangeinterpolation: SUMME(j=1:r+1)[L_j(j)*E(j)]

end


end