import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
%matplotlib inline
%config InlineBacked.figure_fomat ="svg"
import statistics as stats
import numpy
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score 
import statsmodels.api as sm 
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
concentración1 =input("Ingresa la concentración 1: ") 
absorbancia1 = input("Ingresa la absorbancia 1: ")

concentración2 = input("Ingresa la concentración 2: ") 
absorbancia2 = input("Ingresa la absorbancia 2: ")

concentración3 = input("Ingresa la concentración 3: ") 
absorbancia3 = input("Ingresa la absorbancia 3: ")

concentración4 = input("Ingresa la concentración 4: ")
absorbancia4 =input("Ingresa la absorbancia 4: ")

concentración5 = input("Ingresa la concentración 5: ") 
absorbancia5 =input("Ingresa la absorbancia 5: ")

Concentraciones=[concentración1,concentración2,concentración3,concentración4,concentración5]
Absorbancias=[absorbancia1,absorbancia2,absorbancia3,absorbancia4,absorbancia5]
datos={'CONCENTRACIONES': Concentraciones, 'ABSORBANCIAS': Absorbancias}
df= pd.DataFrame({'CONCENTRACIONES': Concentraciones, 'ABSORBANCIAS': Absorbancias})
df
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 3.84))

df.plot(
    x    = 'CONCENTRACIONES',
    y    = 'ABSORBANCIAS',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Concentración vs Absorbancia');
x1=[Concentraciones]
y1=[Absorbancias]
x2 = np.array(x1) 
y2 = np.array(y1) 
n=len(x1)
x_new = x2.astype(float)
y_new=y2.astype(float)
x_mean = np.mean(x_new)
y_mean= np.mean(y_new)
x_mean,y_mean
Sxy = np.sum(x_new*y_new)- 5*x_mean*y_mean 
Sxx = np.sum(x_new*x_new)-5*x_mean*x_mean 
b1 = Sxy/Sxx 
b0 = y_mean-b1*x_mean 
print('La pendiente es:', b1) 
print('El intercepto es: ', b0) 
y_pred = b1 * x_new + b0 
error = y_new - y_pred 
se = np.sum(error**2) 
SSt = np.sum((y_new - y_mean)**2) 
R2 = 1- (se/SSt) 
print('La R al cuadrado es:', R2) 
#CALCULO DE LA CONCENTRACIÓN REAL DE LA MUESTRA POR EL MÉTODO DE LA ECUACIÓN DE LA RECTA EN MUESTRA LIQUIDA#
absorbancia_muestra =float(input("Ingresa la absorbancia de la muestra: ") )
if b0 < 0:
    bfinal=(b0*-1)
else:
    bfinal=(b0)
sum= absorbancia_muestra + bfinal
concen_final= sum/b1
print('La concentración final de la muestra con diluición es:', concen_final) 
#CAlCULO DE LA MUESTRA SIN DILUIR#
fc_dilucion=float(input("Ingresa el factor de dilución:"))
print("Recuerda que el factor de dilución es tu aforo sobre la alicuota que tomaste")
c_final= concen_final*fc_dilucion
print("La concentración final de la muestra sin diluir es: ", c_final)
