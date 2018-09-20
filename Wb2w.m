% Eingangs- und Verbindungsmatrizen sowie Biasvektoren f�r in Net
% vorgegebene  in Gesamtgewichtsvektor net.w_0 zusammenfassen

function w=Wb2w(net,IW,LW,b)

dL=net.dL; % Verz�gerungen zwischen den Schichten
dI=net.dI; % Verz�gerungen der Eing�nge
I=net.I;      % Eingangsmatrizen (hier nur eine Eingangsmatrix der in Schicht 1 einkoppelt)
L_f=net.L_f;  % Vorw�rtsverbindungen
M=net.M; % Anzahl der Schicjten des NN

w=[]; %Gesamtgewichtsvektor


for m=1:M  %Alle Schichten m
    
    %Eingangsgewichte
    if m==1 
        for i=I{m}  %Alle Eingangsmatrizen i die in Schicht m einkoppeln
            for d=dI{m,i}   % alle Verz�gerungen i->m
                w=[w;IW{m,i,d+1}(:)];   %Eingangsgewichtsmatrix zu Gesamtgewichtsvektor hinzuf�gen [Matix(:) = vec(Matric)]
            end
        end
    end

    %Verbindungsgewichte
    for l=L_f{m}  %Alle Schichten l die eine direkte Vrw�rtsverbindung zu Schicht m ahben
        for d=dL{m,l}    % alle Verz�gerungen l->m
            w=[w;LW{m,l,d+1}(:)]; %Verbindungsgewichtsmatrix zu Gesamtgewichtsvektor hinzuf�gen  [Matix(:) = vec(Matric)]
        end
    end
    
    %Biasgewichte
    w=[w;b{m}];     %Biasvektor von Schicht m zum Gesamtgewichtsvektor hunzuf�gen
end