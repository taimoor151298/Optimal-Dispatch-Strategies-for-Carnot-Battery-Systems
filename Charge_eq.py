
def  Charge_eq(x,t,u,DuDx):
from Charger import *
#global pa ca ka A G U T_inf ps cs ks e h D
#All values at T kelvin from engineeringtoolbox

#c = [1;1];
#f = [ks/((1-e)*ps*cs);ka/(e*pa*ca)].*dudx;
#s = [(h/((1-e)*ps*cs))*(u(2)-u(1)); (-G/(pa*e))*dudx(2)+(h/(e*pa*ca))*(u(1)-u(2))+ ((U*D*pi)/(e*pa*ca*A))*(T_inf - u(2)) ];

c = [1, 1];
f = [ka/(pa*ca*e); ks/(ps*cs*(1-e))] .* DuDx;
% f = [0; ks/(ps*cs*(1-e))] .* DuDx;
s = [(  -G/(pa*e))*DuDx(1)+(h/(pa*ca*e))*(u(2)-u(1))+(((U*D*pi)/(pa*ca*e*A))*(T_inf-u(1))); (h/(ps*cs*(1-e)))*(u(1)-u(2))];


end
