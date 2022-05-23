#!/usr/bin/env python
# coding: utf-8

# In[462]:


import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import math

import statistics as stat
from sklearn.neural_network import MLPRegressor


# Seguimos el hilo argumental del [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) para trabajar con el 9.2. No nos servirán las variables del ejercicio 9.1, ya que 
# 
# no dividimos el conjunto de datos  en variables independientes y objetivos. Es decir, no vamos a aprobechar nada de él pero si 
# 
# alguna información, que vamos a pasar a resumir:
# 
# 

# In[556]:


df = pd.read_csv("DelayedFlights.csv") # este es el conjunto de datos proporcionado en el ejercicio 
df


# - Las variables DepTime','CRSDepTime', 'ArrTime', 'CRSArrTime" básicamente son las horas programadas y reales que acaban determinando 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay','DepDelay'. Restar  'ArrTime' y 'CRSArrTime nos da ArrDelay, por ejemplo. Mientras en el otro ejercicio, no nos hicieron falta, para entrenar una red neuronal o un arbol, nos pueden venir bien si no contamos con la variable DepDelay( Ejercicio 3)
# 
# - Las variables "CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay" tienen un tercio de valores NaN, y la suma de todas es el retraso general. 
# 
# - 'FlightNum' y'TailNum' son variables catogóricas que no aportan mucho 
# 
# - "ActualElapsedTime",  "CRSElapsedTime","AirTime" y "Distance" tienen un coeficiente de correlación lineal entre ellas cercano a 0,95 $\pm 2$ , mientras que ArrDelay y DepDelay, tienen una correlación cercana a cero respecto a estas 4 variables. 
# 
# - Arrdelay y DepDelay tienen una correlación cercana a 0,8. 
# 
# - Entre el primer grupo(distance) y el segundo(arrdelay) el Coeficiente de correlación tiende a cero
# 
# 
# - los aeropuertos donde hay más vuelos hay más minutos acumulados de retrasos. No crece de manera lineal, va creciendo de forma $ax^2 +bx$ de forma muy aplanada. Ésto estaba por corroborar. 
# 
# - Con las compañías crece de forma lineal. A más vuelos, más minutos acumulados de retrasos. 
# 
# - Ignoraré las variables Cancelled o Diverted ya que si tienen un valor de 1 ( y no llegan a 8000 casos) tienen valores NaN en ArrDelay y DepDelay
# 
# 
# 
# 

# 0. **PREPROCESAMIENTO DE DATOS.** 
# 
# Vamos a dedicar esta parte a tratar las variables con valores perdidos, transformación de estas variables, selección, y otras 
# 
# manipulaciones para poder aplicar a las regresiones 

# In[464]:


df.columns


# 0.1. **Muestreo de población y tratamiento de valores perdidos**
# 
# Primero hacemos un **muestreo de la población** para evitar problemas de memoria, ya que al hacer ciertas operaciones
# 
# sobre todo el conjunto de datos, el ordenador da problemas de memoria, incluso vectorizando 
# 

# In[557]:


df1= df.sample(390000)# el 20% aproximadamente 
df1.isnull().sum()# miramos la cantidad de valores perdidos


# Como hicimos en la [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) vamos a ver si estos valores NaN corresponden a vuelos cancelados o derivados, que no van a servir de nada para calcular el valor de ArrDelay

# In[562]:


# vamos a ver si todos los valores NaN corresponden a todos lo vuelos Cancelados o Derivados 
df1_div= df1[df1["Diverted"]==1]
df1_can=df1[df1["Cancelled"]==1]
df1_div.shape#número de vuelos desviados


# In[563]:


df1_can.shape#número de vuelos cancelados


# In[564]:


df1_div.shape[0]+df1_can.shape[0]# número total de cancelados y derivados. 


# In[565]:


df1_div["ArrTime"].isnull().sum()+df1_can["ArrTime"].isnull().sum()# que coincide con los valores NaN de ArrTime


# In[566]:


df1_div["ActualElapsedTime"].isnull().sum()+df1_can["ActualElapsedTime"].isnull().sum()# esto daría lo mismos 
# para ArrDelay y DepDelay


# Los valores NaN, coinciden con aquellos que el vuelos ha sido Cancelado o Desviado,por lo tanto, no aportan información 
# 
# al cálculo de la regresión de ArrDelay, optamos por eliminarlos

# In[567]:


df2=df1.dropna( subset=["ArrDelay"]).reset_index(drop=True)
df2.isna().sum()


# 0.2. **Transformación de variables horarias**
# 
# En esta parte vamos a convertir las variables DepTime,	CRSDepTime,	ArrTime,	CRSArrTime en la función cíclica. 
# 
# Estas cuatro variables vienen en formato horario  hh:mm. Lo que haremos será contar todos los minutos transcurridos durante 
# 
# el día, siendo 0 minutos a las 00:00 y 1440 los minutos transcurridos durante el día a las 23:59.
# 
# Más adelante, en el apartado 0.4, transformaremos estas variables en cíclicas. 
# 
# 
# 
# 
# 
# 

# In[568]:


# Primero de todo convierto las variables horarias en formato hora y para eso tienen que haber 4 digítos, que los relleno por la
#izquierda con ceros
# Primero tengo que convertir en entero las variables DepTime y ArrTime en enteros para evitar los decimales


df2['DepTime'] = df2['DepTime'].astype(int)


# In[569]:



df2['ArrTime'] = df2['ArrTime'].astype(int)
df2


# In[570]:


#relleno por la izquierda con ceros
df2['DepTime'] = df2['DepTime'].astype(str).str.zfill(4)
df2['CRSDepTime'] = df2['CRSDepTime'].astype(str).str.zfill(4)
df2['ArrTime'] = df2['ArrTime'].astype(str).str .zfill(4)
df2['CRSArrTime'] = df2['CRSArrTime'].astype(str).str.zfill(4)
df2.head()


# In[571]:


# las convierto en formato horario( Nota: en un principio lo pasé a formato horario por si lo necesitaba para datetime, pero 
# al final opté por otro tipo de conversión)
df2['DepTime'] = df2['DepTime'].astype(str).str[:2]  + ':' + df2['DepTime'].astype(str).str[2:4] + ':00' 
df2['CRSDepTime'] = df2['CRSDepTime'].astype(str).str[:2] + ':' + df2['CRSDepTime'].astype(str).str[2:4] + ':00' 
df2['ArrTime'] = df2['ArrTime'].astype(str).str[:2] + ':' + df2['ArrTime'].astype(str).str[2:4]  + ':00'
df2['CRSArrTime'] = df2['CRSArrTime'].astype(str).str[:2] + ':' + df2['CRSArrTime'].astype(str).str[2:4] + ':00'


df2


# In[572]:


# creamos la función minutos, que divide la hora hh:mm con un Split, en una lista ("hh","mm"), reconvierte hh y mm en enteros,
# para luego pasarlos a minutos, y con la reconverión ya comentada aplica la función minutos()
def minutos(x):    
    x=x.split( sep=":")
    seg= 60*(int(x[0]))+(int(x[1]))
    
    return seg



dfhoras= df2[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]]

    


# In[573]:


dfhoras_DT= dfhoras["DepTime"].apply(minutos)
dfhoras_CRSD=dfhoras["CRSDepTime"].apply(minutos)
dfhoras_AT=dfhoras["ArrTime"].apply(minutos)
dfhoras_CRSA=dfhoras["CRSArrTime"].apply(minutos)


# In[574]:


df2.columns


# In[575]:


df3= df2.drop(['Unnamed: 0', 'Year', 'DepTime',
       'CRSDepTime', 'ArrTime', 'CRSArrTime','FlightNum', 'TailNum', 'Cancelled', 'CancellationCode', "Diverted"], axis=1) 
# 'Unnamed: 0' y  'Year" no aportan nada,  TailNum y Flight Num son eliminadas por cuestión de practicidad y para no 
# añadir columnas vía Dummies por cada código de vuelo o Tailnum 
# Y al Final , Cancelled y Diverted no aportan nada al cálculo de arrdelay


# In[576]:


# ahora añadimos las cuatro columnas nuevas

df4= pd.concat([df3, dfhoras_DT,dfhoras_CRSD , dfhoras_AT, dfhoras_CRSA], axis=1)
df4.columns


# In[578]:


df4[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]].describe()# miramos como quedan para ver si hay alguna anomalía en 
# los máximos y mínimos


# In[579]:


df4.isna().sum()#revisamos el número de nulos


# 0.3 **Transformación de Variables Delay.**
# 
# Vamos a analizar la variables ArrDelay y DepDelay, rescepto a los motivos del retraso. 
# 
# 

# In[580]:


df_delay= df4[['ArrDelay','DepDelay','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay',]]

df_delay


# A simple vista vemos que la suma de minutos por fila  de CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
# 
# 'LateAircraftDelay', es igual a ArrDelay 

# In[581]:


sns.pairplot( df_delay,kind="reg", x_vars=['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay'], y_vars="ArrDelay")


# In[582]:


# Tras visualizar el DF veo como posibilidad que si ArrDelay es menor que x minutos ( x podría ser de 10 o 20 minutos ), 
# las X_vars tienen valores NaN
# también veo que si se da la condición(min< x), las cinco variables de X_Vars son NaN
df_delay[df_delay["CarrierDelay"].isna()]


# In[583]:


df_delay[df_delay["ArrDelay"]>=20].isna().sum()#miramos varios valores de Arrdelay para ver cuando empiezan a parecer valores NaN


# In[584]:


df_delay[df_delay["ArrDelay"]>18].isna().sum()


# In[585]:


df_delay[df_delay["ArrDelay"]>15].isna().sum()


# In[586]:


df_delay[df_delay["ArrDelay"]>13].isna().sum()


# In[591]:


dfd=df_delay[df_delay["ArrDelay"]>=15]# de manera definitiva
dfd.isna().sum()


# In[592]:


dfd=df_delay[df_delay["ArrDelay"]<15]
dfd.isna().sum()


# Con esto acabamos de demostrar que los valores de **ArrDelay < 15 minutos**, no registran las causas de retraso.
# 
# Podriamos sustituir los valores NaN repartiendo lo minutos de Arrdelay<15 entre las cinco variables, pero hay valores de Arrdelay negativos. **Así que asignaremos un valor de 0.0 a todos lo NaN**
# 
# 

# In[593]:


dfd2=df_delay[df_delay["ArrDelay"]>=15]
dfd2.isna().value_counts()


# In[595]:


delay_not_NAN= df_delay[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay']].fillna(0.0)
delay_not_NAN.isna().value_counts()


# In[596]:


delay_not_NAN.describe()


# In[597]:


df5_0= df4.drop(  ["CarrierDelay", 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay'], axis=1 )# borramos los 
#valores con NaN, y añadimos las variables con la sustitución de NaN por cero
df5= pd.concat([df5_0, delay_not_NAN], axis =1)
df5.isna().sum()


# 0.4 **Variables Categóricas y Visualización de variables.** 
# 
#  En este apartado vamos a ver de que manera se comportan las viarbles para determinar cuales son independientes o no, así como 
#  analizar la categóricas. 
#  
# 

# In[598]:


df5.columns


# Cómo recordamos en el [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git), habían 300 **origenes y destinos**, así como 20 **compañías**. El retraso acumulado en ArrDelay 
# 
# tenía una relación lineal en el número de vuelos por compañías, así que tampoco aportaba mucha más información. 
# 
# Mientras que por aeropuerto tenía una tendencia lineal, y crecía algo más para aeropuertos con muchos vuelos 
# 
# Así que hacer un get_dummies nos generaría 640 variables, así que **vamos a eliminarlas**. 

# In[599]:


df5.drop(["UniqueCarrier", "Origin", "Dest"], axis=1, inplace=True)


# In[600]:


# Vamos a ver la correlación lineal 
df5corr= df5.corr()
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df5corr, vmax=1, vmin=-1, square=True,cmap="Blues", annot=True)
plt.show()

#


# Nos permite ver que LateAircraft Delay, NasDelay, WheatherDelay y Depdelay son las variables con alta correlación lineal con ArrDelay, y como vimos en el 9.1 ActualElapsedTime', 'CRSElapsedTime', 'AirTime', y Distance  están altamente correlaiconadas
# entre ellas pero no con ArrDelay
# 
# Las variables temporales (mes, dia del mes y dia de la semana ) tienen una correlación casi nula con el resto de 
# variables, además de que en este DataSet funcionan como variables categóricas. 
# 
# Empecemos por ver como se distribuye arrdelay en función de los meses, que cómo ya habíamos visto en el ejercicio 9.1 hay más 
# vuelos de Diciembre a Marzo y de Junio a Agosto. 
# 
# También, si nos fijamos en los días de la semana, vimos en 9.1 que los Jueves, Viernes, Domingos y Lunes tenían más vuelos que los otros días. Siendo el Viernes el día con más vuelos y el sábado el que menos. 

# In[601]:


df5ArrMonth= df5[df5["ArrDelay"]<200] # miramos como se distribuye Arrdelay en función del mes
order= []
for i in range(1,13):
    order.append(i)

g = sns.FacetGrid(df5ArrMonth, row="Month", row_order= order,
                  height=2.7, aspect=5,)
g.map( sns.histplot, "ArrDelay", stat="count")


# In[602]:


df5[["ArrDelay", "Month"]].groupby("Month").mean()


# In[603]:


df5ArrMonth= df5[df5["ArrDelay"]<200] # y hacemos lo mismo para el día de la semana 
order2= []
for i in range(1,8):
    order2.append(i)

g = sns.FacetGrid(df5ArrMonth, row="DayOfWeek", row_order= order2,
                  height=2.7, aspect=5,)
g.map( sns.histplot, "ArrDelay", stat="count")


# In[604]:


df5[["ArrDelay", "DayOfWeek"]].groupby("DayOfWeek").mean()


# In[605]:


#También mmiramos la distribución del retraso acumulado por meses  y días  

df5[["ArrDelay", "Month"]].groupby("Month").sum().plot(kind="bar") 
df5[["ArrDelay", "DayOfWeek"]].groupby("DayOfWeek").sum().plot(kind="bar")


# Las últimas distribuciones tienen una distribución casi idéntica al conteo  de vuelos de Month y Dayofweek que vimos  
# en [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git); es decir, a más vuelos, más retraso acumulado. 
# 
# Teniendo en cuenta que los vuelos pueden llegar al día siguiente, como los vuelos salidos el Lunes pueden llegar el Martes,
# la media de atrasos en función de los días de la semana se mantiene en $ 41.5 \pm 2.5 $. Teniendo en cuenta ésto, y que el día 
# de la semana tiene un índice de correlación muy bajo, y que como pasaba con las compañías, hay más atrasado acumulado en función del dia de la semana o del mes a causa de que hay más cantidad de vuelos. Vamos a descartar estas dos variables. 
# 
# No voy a hacer el mismo razonamiento para DayofMonth, ya que puedo usar el mismo argumento que con las otras dos variables.
# 
# El día del Mes, el mes y el día de la semana , no influye en los retrasos y además nos generaría contabilizar muchas más columnas, ya que son variables categóricas. 
# 

# In[606]:


df6= df5.drop(['Month', 'DayofMonth', 'DayOfWeek'], axis=1)


# In[608]:


df6.shape# miramos dimensiones del Data Set


# Las Variables "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime", deberían ser una función cíclica, es decir, cuando las horas de 
# Arrtime sea menor que DepTime, habría contar 1440 minutos más, dando igual si es CRS o no  

# In[508]:


df6[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]].describe()


# In[609]:


f,axs= plt.subplots(2, 2, figsize=(14,8))
sns.regplot(data=df6, x=  "DepTime",y = "ArrDelay",ax= axs[0,0], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

sns.regplot( data=df6, x=  "CRSDepTime",y = "ArrDelay",ax= axs[0,1], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})  

sns.regplot( data=df6, x=  "ArrTime",y = "ArrDelay",ax= axs[1,0]  , ci=None, scatter_kws={"color": "blue"}, line_kws={"color": "red"})

sns.regplot( data=df6, x=  "CRSArrTime",y = "ArrDelay",ax= axs[1,1]  , ci=None, scatter_kws={"color": "blue"}, line_kws={"color": "red"})


# In[614]:


minimo=df[df["ArrDelay"]<0].sort_values("ArrDelay")# miramos los valores más bajos de ArrDelay
zmin=minimo["ArrDelay"].min()
zmin


# In[615]:


# podemos observar una independencia casi total, pero si hacemos lo siguiente. 
def rest(z):
    x=z[0]
    y=z[1]
    if (x < y) & ((x-y)<zmin): 
        t= (1440+x)-y
        return t
    else:
        t= x-y
        return t  
    
x11= df6[["DepTime","CRSDepTime" ]].apply(rest, axis=1)
x10= df6[["ArrTime","CRSArrTime" ]].apply(rest,axis=1)

f,axs= plt.subplots(2, 1, figsize=(14,8))

sns.regplot(data=df6, x=  x10,y = "ArrDelay",ax= axs[0], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

sns.regplot( data=df6, x=  x11,y = "DepDelay",ax= axs[1], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})  

plt.show()


# Sean **x10** y **x11** definidas por 
# 
# **x10= df6["ArrTime"]-df6["CRSArrTime"]**
# 
# **x11= df6["DepTime"]-df6["CRSDepTime"]**
# 
# podemos observar una relación lineal.
# 
# 
# 
# Por otro lado, habíamos visto en el ejercicio 9.1 lo siguiente:
# 
# "ActualElapsedTime", "CRSElapsedTime","AirTime" y "Distance" tienen un coeficiente de correlación lineal entre ellas cercano a 0,95  ±2  , mientras que ArrDelay y DepDelay, tienen una correlación cercana a cero respecto a estas 4 variables.
# 
# Arrdelay y DepDelay tienen una correlación cercana a 0,8.
# 
# Por lo que sólo nos queda estudiar Taxiin y taxiout

# In[616]:


f,axs= plt.subplots(2, 1, figsize=(14,8))

sns.regplot(data=df6, x=  "TaxiIn",y = "ArrDelay",ax= axs[0], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

sns.regplot( data=df6, x=  "TaxiOut",y = "ArrDelay",ax= axs[1], ci=None,scatter_kws={"color": "blue"}, line_kws={"color": "red"})  

plt.show()


# Podemos ver que ambas variables (TaxiIn y TaxiOut)tienen una correlación muy baja respecto a ArrDelay, a pesar de la tendencia 
# lineal. Pero hay una gran acumulación de restrasos importantes a pesar de tiempos cortos de tiempos de despegue (TaxiOut) o de 
# aterrizaje, lo que hace perder la linealidad para valores bajos de taxiin o taxiouit
# 
# 

# 0.5 **Transformación y Selección de Variable** 
# 
# Vamos a usar el coeficiente de correlación para seleccionar las características que mejor nos sirven para la regressión. 
# 
# Antes de todo, vamos a remodelar el data set, añadiendo x10 y x11, y eliminado 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', ya que x10 y x11 son combinaciones lienales de estas 4. 
# 
# También eliminamos 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime' ya que tienen un 0.95 de correlación o más  con Distance y aportan la misma información. 
# 
# 

# In[617]:


x10=x10.rename("X10")
x11=x11.rename("X11")
x10.describe()


# In[618]:


x11.describe()


# In[619]:


x1= pd.concat([x10,x11,df6["ArrDelay"]],axis=1)
x1.corr()


# In[620]:


#remodelamos el DF con x10 y x11
df6b=df6.drop(['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime'], axis=1)
df7=pd.concat([df6b, x10,x11], axis =1)
df7


# -Hasta el momento hemos reducido una docena de variables categóricas, bien porque no aportaban nada como Year, bien porque 
# nos aumentaban la dimensionalidad del DataSet  como Origen o bien porque no influenciaban en nada  como 
# el mes en que se daba el vuelo sobre el ArrDelay.
# 
# -Hemos reducido 4 variables continuas a x10 y x11
# 
# -Y hemos reducido 4 variables a 1, debido a su alto grado de correlación entre ellas( dejando Distance)
# 
# Con las que quedan, vamos a ver el coeficente de correlación de Spearman, pero antes de todo,vamos a estandarizar las variables 
# 
# 

# In[622]:


scaler = StandardScaler()

df8= pd.DataFrame(scaler.fit_transform(df7), columns=df7.columns )

df8.describe().round(3)


# In[623]:


# y aplicamos el coeficiente de correlación de Spearman
df8corr= df8.corr(method="spearman")
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df8corr, vmax=1, vmin=-1, square=True,cmap="Blues", annot=True)
plt.show()


# EL coeficiente de correlación de Sperman tampoco no aporta mucho más que el de Pearson. Vamos a mirar un método de selección 
# de variables, para ver cuales son las mejores variables. Escogeremos f_regression.

# In[624]:


df8.columns


# In[626]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
Xpreseleccion= df8.drop(["ArrDelay"], axis=1)
ypreseleccion= df8["ArrDelay"]
for i in range(1,12): # vamos a mirar como f_selection va escogiendo las variables en función del estadístico F
    fs = SelectKBest(score_func=f_regression, k=i)
    XS=fs.fit(Xpreseleccion,ypreseleccion)
    

    filter=fs.get_support()
    variables=np.array(Xpreseleccion.columns)
    print(variables[filter])
    print (XS.scores_[filter])
    print ("\n")




# Observamos que la selección de variables por f_Regression nos lleva a unos resultados muy similares al coeficiente correlación 
# Pearson. Así que escogeremos la variables con mayor inidce que correlación:
#     
#  sean las variables,    ['DepDelay' 'TaxiOut' 'CarrierDelay' 'NASDelay' 'LateAircraftDelay']
#     

# In[627]:


df9= df8[["ArrDelay",'DepDelay', 'CarrierDelay', 'NASDelay' ,'LateAircraftDelay', 'X10', 'X11']]
df9.head()


# 0.6 **Separación entre Train y Test**. 

# In[669]:


X_= df9.drop(["ArrDelay"], axis=1)

y_= df9["ArrDelay"]

X_train, X_test, y_train, y_test = train_test_split( X_, y_, test_size=0.30, random_state=42, shuffle=True)


# 1.  **Modelos de regresión**
# 
# Crea tres modelos de regresión para predecir ArrDelay 

# In[670]:


#Creamos los 3 modelos
lin = LinearRegression()
rft=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42 )
mpl= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="sgd",max_iter=300 )


# - **Modelo de regresión lineal**

# In[671]:



lin.fit(X_train, y_train)
fx_lin= lin.predict(X_test)# la f(x0,....,xn)predicha

print(fx_lin)# valores de fx predichos
print(fx_lin.shape) #tamaño de la matriz


# In[672]:


r_lin = lin.score(X_test, y_test)
r_lin # podemos ver que el ajuste de correlación es de valor alto


# In[673]:


# sea y = A1X1+A2X2+......+AnXn + b
print('punto de corte del eje y(x=0) ', lin.intercept_) # marca el punto de intercepeción del eje y cuando x=0, b
print( "coeficientes de las variables x", lin.coef_)# marca los coeficientes A1,...:AN de las variables indeondientes 


# - **Modelo de bosque aleatorio** 

# In[674]:


rft.fit(X_train, y_train)
fx_rft= rft.predict(X_test)# la f(x0,....,xn)predicha
print(fx_rft)# valores de fx predichos


# In[675]:


# miramos la correlación lienal
r_rft = rft.score(X_test, y_test)
r_rft


# In[676]:


# como no calcula coeficientes, ya que los cálculos los hace en función de la ganancia de información de cada arbol del conjunto
# aleatorio, miramos la importancia de cada caracterísitca. 
importance = rft.feature_importances_
variables_finales=['DepDelay', 'CarrierDelay', 'NASDelay' ,'LateAircraftDelay', 'X10', 'X11']
for i,j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,j), variables_finales[i])


# - **Modelo de red neuronal perceptron multicapa**

# In[677]:


mpl.fit(X_train, y_train)
fx_mpl= mpl.predict(X_test)# la f(x0,....,xn)predicha

print(fx_mpl)# valores de fx predichos


# In[678]:


# comprobamos el coeficiente de la regresión
r_mpl= mpl.score(X_test, y_test)
r_mpl


# 2. **Comprueba el error en la métrica de MSE y R2**
# 

# Podemos observar, que los coeficientes de correlación tienen muy buenos ajustes. 
# 
# 

# In[679]:


print( "correlación lineal para la regresión lineal ", r_lin)


# In[680]:


print( "correlación lineal para el bosque aleatorio ", r_rft)


# In[681]:


print( "correlación lineal para el perceptrón multicapa ", r_mpl)


# Ahora miramos el error cuadrático medio

# In[703]:


print('Error cuadrático medio de la regresión lineal :', mean_squared_error(y_test, fx_lin)) 


# In[704]:


print('Error cuadrático medio del bosque aleatorio:', mean_squared_error(y_test, fx_rft)) 


# In[705]:


print('Error cuadrático medio de la red enurona:', mean_squared_error(y_test, fx_mpl))


# Contando que las variables están estandarizadas, y la media tiende a cero, y el error cuadrático medio reduce las 
# discrepancias menores a uno, el que mejor rendimiento saca es el Bosque Aleatorio

# - **Ejercicio 3** :Modificar el modelo para ver el resultado. 
# 
# Vamos a modificar ciertos parámetros del modelo para ver cómo responden

# ------------------------------------------------------------------------------------

# .  **Regresión lineal** 

# Modificamos el témino constante 

# In[686]:


#Empecemos modificando, quitando el término constante de la ecuación
lin2 = LinearRegression(fit_intercept= False  )


# In[687]:


lin2.fit(X_train, y_train)
fx_lin2= lin2.predict(X_test)


# In[701]:


r_lin2 = lin2.score(X_test, y_test)
print("coeficiente sin término constante", r_lin2)
print("coeficiente con término constante ", r_lin) 
print("diferencia de ajuste  = ", (r_lin-r_lin2))
if r_lin< r_lin2: 
    print ( "la regresión sin término constante tiene mejor ajuste ")
else: 
    print ( "la regresión con término constante tiene mejor ajuste ")
    


# In[707]:


print( "coeficientes de las variables x sin término constante ", lin2.coef_)
print( "coeficientes de las variables x con término constante ", lin.coef_)


# In[759]:


print('Error cuadrático medio de la regresión lineal sin término constante:', mean_squared_error(y_test, fx_lin2))
print('Error cuadrático medio de la regresión lineal con término constante:', mean_squared_error(y_test, fx_lin))
if  mean_squared_error(y_test, fx_lin)<  mean_squared_error(y_test, fx_lin2): 
    print ( "la regresión con término constante tiene menor error de ajuste ")
else: 
    print ( "la regresión sin término constante tiene menor error de ajuste ")
    


# Tenemos unas discrepancias mínimas, podemos ver que el ajuste sin término constante tiene mejor ajuste con el coeficiente de 
# correlación, sin embargo su error es menor en el término constante 

# . **Bosque Aleatorio.** 

# Empecemos modificando el número de árboles a la mitad

# In[714]:


rft2=RandomForestRegressor( n_estimators=100, max_depth=10, random_state=42 )


# In[715]:


rft2.fit(X_train, y_train)
fx_rft2= rft2.predict(X_test)


# In[722]:


r_rft2 = rft2.score(X_test, y_test)
print("r del bosque  con 100 árboles  ", r_rft2)
print("r del bosque  con 200 árboles  ", r_rft) 


# In[723]:


if r_rft< r_rft2: 
    print ( "bosque  con 100 árboles, tiene mejor ajuste")
else: 
    print ( "bosque  con 200 árboles , tiene mejor ajuste ")


# In[760]:


print('Error cuadrático medio de la regresión lineal con 100 árboles:', mean_squared_error(y_test, fx_rft2))
print('Error cuadrático medio de la regresión lineal con 200 árboles:', mean_squared_error(y_test, fx_rft))
if  mean_squared_error(y_test, fx_rft)<  mean_squared_error(y_test, fx_rft2): 
    print ( "la regresión con 200 árboles tiene menor error de ajuste ")
else: 
    print ( "la regresión con 100 árboles tiene menor error de ajuste ")


# modificamos el máximo de la profundidad de las capas a la mitad

# In[719]:


rft3=RandomForestRegressor( n_estimators=200, max_depth=5, random_state=42 )


# In[720]:


rft3.fit(X_train, y_train)
fx_rft3= rft3.predict(X_test)


# In[725]:


r_rft3 = rft3.score(X_test, y_test)
print("r del bosque  con 5 capas   ", r_rft3)
print("r del bosque  con 10 capas   ", r_rft) 


# In[726]:


if r_rft< r_rft3: 
    print ( "bosque  con 5 capas, tiene mejor ajuste")
else: 
    print ( "bosque con 10 capas , tiene mejor ajuste ")


# In[761]:


print('Error cuadrático medio de la regresión lineal con 5 capas:', mean_squared_error(y_test, fx_rft3))
print('Error cuadrático medio de la regresión lineal con 10 capas:', mean_squared_error(y_test, fx_rft))
if  mean_squared_error(y_test, fx_rft)<  mean_squared_error(y_test, fx_rft3): 
    print ( "la regresión con 10 capas tiene menor error de ajuste ")
else: 
    print ( "la regresión con  5 capas tiene menor error de ajuste ")


# - **Red Neuronal**

# Vamos a modificar el número de capas, la tasa de aprendizaje, la función de activación, el método de cálculo del gradiente, 

# In[728]:


mpl1= MLPRegressor(hidden_layer_sizes=(20,20), activation="logistic", solver="sgd",max_iter=300 )# número de capas
mpl2= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="relu", solver="sgd",max_iter=300 )# la función de activación
mpl3= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="sgd",max_iter=300, learning_rate_init=0.01, 
                  learning_rate= "invscaling")# modificando la tasa de aprendizaje  
mpl4= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="adam",max_iter=300 )# solventamos por Adam


# En MPL1 hemos quitado una capa 

# In[729]:


mpl1.fit(X_train, y_train)#Entrenamos los modelos


# En MPL2 hemos cambiado la función de activación tipo RELU, tal que si x<0 Relu=0 , si x>=0, Relu=x

# In[730]:


mpl2.fit(X_train, y_train)


# En MPL 3 hemos modificado hemos modificado la tasa de aprendizaje de 0.001 a 0.01, y en vez de ser constante será decreciente. 

# In[731]:


mpl3.fit(X_train, y_train)


# En mpl4 hemos cambiado la manera de solventart el cálculo de pesos, pasando de una solución por gradiente a una por gradiente 
# en que la tasa de aprendizaje es adaptativa(ADAM)

# In[732]:


mpl4.fit(X_train, y_train)


# **Red con dos capas en vez de tres**

# In[736]:


# Analizamos la red con una capa menos 
fx_mpl1= mpl1.predict(X_test)


# In[737]:


r_mpl1 = mpl1.score(X_test, y_test)
print("r de red con dos capas  ", r_mpl1)
print("r de red  con 3 capas   ", r_mpl)
if r_mpl< r_mpl1: 
     print ( "red con dos capas ocultas tiene mejor ajuste")
else: 
    print ( "red con tres capas ocultas tiene mejor ajuste")
 


# In[762]:


print('Error cuadrático medio de la regresión lineal con 2 capas:', mean_squared_error(y_test, fx_mpl1))
print('Error cuadrático medio de la regresión lineal con 3 capas:', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(y_test, fx_mpl1): 
    print ( "la regresión con 3 capas tiene menor error de ajuste ")
else: 
    print ( "la regresión con  2 capas tiene menor error de ajuste ")


# Podemos ver que la red de 3 capas  tiene un mejor ajuste y menor error

# ---------------------------------------------

# - **Red con función de activación RELU**

# In[739]:


fx_mpl2= mpl2.predict(X_test)


# In[741]:


r_mpl2 = mpl2.score(X_test, y_test)
print("r de red con función de activación RELU  ", r_mpl2)
print("r de red  con función de activación logística  ", r_mpl)
if r_mpl< r_mpl2: 
     print ( "red con función de activación RELU tiene mejor ajuste")
else: 
    print ( "red con función de activación logística tiene mejor ajuste")
 


# In[763]:


print('Error cuadrático medio de la regresión lineal con función de activación Relu:', mean_squared_error(y_test, fx_mpl2))
print('Error cuadrático medio de la regresión lineal con función de activación logísitca:', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(y_test, fx_mpl2): 
    print ( "la regresión con función de activación logísitca tiene menor error de ajuste ")
else: 
    print ( "la regresión con  función de activación Relu: tiene menor error de ajuste ")


# - **red con tasa de aprendizaje descendiente** 

# In[743]:


fx_mpl3= mpl3.predict(X_test)


# In[749]:


r_mpl3 = mpl3.score(X_test, y_test)
print("r de red con tasa de aprendizaje descendiente  ", r_mpl3)
print("r de red  con tasa de aprendizaje constante  ", r_mpl)
if r_mpl< r_mpl3: 
     print ( "red con tasa de aprendizaje descendiente tiene mejor ajuste")
else: 
    print ( "red con tasa de aprendizaje constante tiene mejor ajuste")
 


# In[764]:


print('Error cuadrático medio de la regresión lineal con tasa de aprendizaje descendiente:', mean_squared_error(y_test, fx_mpl3))
print('Error cuadrático medio de la regresión lineal con tasa de aprendizaje constante :', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(y_test, fx_mpl3): 
    print ( "la regresión con tasa de aprendizaje constante tiene menor error de ajuste ")
else: 
    print ( "la regresión con  tasa de aprendizaje descendiente tiene menor error de ajuste ")


# Con estos dos cambios, pasar de la tasa de aprendizaje a 0.01 y hacerla descendiente, hemos perdido calidad en el ajuste y en el 
# error

# - **Red que calcula los pesos con el método de adam**

# In[748]:


fx_mpl4= mpl4.predict(X_test)


# In[752]:


r_mpl4 = mpl4.score(X_test, y_test)
print("r de red con método de ADAM  ", r_mpl4)
print("r de red  con método de SGD  ", r_mpl)
if r_mpl< r_mpl4: 
     print ( "red con método de ADAM tiene mejor ajuste")
else: 
    print ( "red con método de SGD tiene mejor ajuste")
 


# In[765]:


print('Error cuadrático medio de la regresión lineal con método de ADAM:', mean_squared_error(y_test, fx_mpl4))
print('Error cuadrático medio de la regresión lineal con método de SGD :', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(y_test, fx_mpl4): 
    print ( "la regresión  con método de SGD tiene menor error de ajuste ")
else: 
    print ( "la regresión con método de ADAM tiene menor error de ajuste ")


# ------------------------------------------------------------------------

# - **Ejercicio 4.**  Validación Interna. 
# 
# Vamos a ha hacer las mismas pruebas pero sin hacer entrenamiento 

# In[753]:


linvi = LinearRegression()
rftvi=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42 )
mplvi= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="sgd",max_iter=300 )


# - **Regresión lineal** 

# In[754]:


linvi.fit(X_, y_)
fx_linvi= linvi.predict(X_)


# In[816]:


r_linvi = linvi.score(X_, y_)
print("r de regresión lineal sin entrenamiento  ", r_linvi)
print("r de regresión lineal con entrenamiento ", r_lin)
if r_lin< r_linvi: 
     print ( "regresion sin entrenamiento tiene  mejor ajuste")
else: 
    print ( "regresion con entrenamiento tiene mejor ajuste")


# In[766]:


print('Error cuadrático medio de la regresión lineal sin entrenamiento :', mean_squared_error(y_, fx_linvi))
print('Error cuadrático medio de la regresión lineal con entrenamiento :', mean_squared_error(y_test, fx_lin))
if  mean_squared_error(y_test, fx_lin)<  mean_squared_error(y_, fx_linvi): 
    print ( "la regresión  con entrenamiento tiene menor error de ajuste ")
else: 
    print ( "la regresión sin entrenar tiene menor error de ajuste ")


# - **Bosque aleatorio** 

# In[768]:


rftvi.fit(X_, y_)
fx_rftvi= rftvi.predict(X_)


# In[817]:


r_rftvi = rftvi.score(X_, y_)
print("r de regresión lineal sin entrenamiento  ", r_rftvi)
print("r de regresión lineal con entrenamiento ", r_rft)
if r_rft< r_rftvi: 
     print ( "bosque aleatorio sin entrenamiento tiene mejor ajuste")
else: 
    print ( "bosque aleatorio con entrenamiento tiene mejor ajuste")


# In[773]:


print('Error cuadrático medio de la regresión lineal sin entrenamiento :', mean_squared_error(y_, fx_rftvi))
print('Error cuadrático medio de la regresión lineal con entrenamiento :', mean_squared_error(y_test, fx_rft))
if  mean_squared_error(y_test, fx_rft)<  mean_squared_error(y_, fx_rftvi): 
    print ( "la regresión  con entrenamiento tiene menor error de ajuste ")
else: 
    print ( "la regresión sin entrenar tiene menor error de ajuste ")


# - **Red Neuronal** 

# In[774]:


mplvi.fit(X_, y_)
fx_mplvi= mplvi.predict(X_)


# In[783]:


r_mplvi = mplvi.score(X_, y_)
print("r de regresión lineal sin entrenamiento  ", r_mplvi)
print("r  de regresión lineal con entrenamiento ", r_mpl)
if r_mpl< r_mplvi: 
     print ( "red sin entrenamiento tiene mejor ajuste")
else: 
    print ( "red con entrenamiento tiene mejor ajuste")


# In[776]:


print('Error cuadrático medio de la regresión lineal sin entrenamiento :', mean_squared_error(y_, fx_mplvi))
print('Error cuadrático medio de la regresión lineal con entrenamiento :', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(y_, fx_mplvi): 
    print ( "la regresión  con entrenamiento tiene menor error de ajuste ")
else: 
    print ( "la regresión sin entrenar tiene menor error de ajuste ")


# **NIVEL 2.** 
# 
# "Realiza  algún proceso de ingeniería para mejorar el proceso." 
# 
# En verdad, esto ya lo he hecho en el apartado 0, preproceso. 
# 
# Así que vamos intentar ver que pasa si en vez de Estandarizar, normalizamos 

# In[798]:


from sklearn.preprocessing import Normalizer
norm= Normalizer()
df8norm= pd.DataFrame(norm.fit_transform(df7), columns=df7.columns )
df8norm.head(10)


# In[799]:


df9norm= df8norm[["ArrDelay",'DepDelay', 'CarrierDelay', 'NASDelay' ,'LateAircraftDelay', 'X10', 'X11']]
Xn= df9norm.drop(["ArrDelay"], axis=1)

yn= df9norm["ArrDelay"]

Xntrain, Xntest, yntrain, yntest = train_test_split( Xn, yn, test_size=0.30, random_state=42, shuffle=True)


# In[800]:


linn = LinearRegression()
rftn=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42 )
mpln= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="sgd",max_iter=300 )


# - **REGRESIÓN LINEAL**

# In[801]:


linn.fit(Xntrain, yntrain)
fx_linn= linn.predict(Xntest)


# In[804]:


r_linn = linn.score(Xntest, yntest)
print("r de regresión lineal  sobre variables normalizadas  ", r_linn)
print("r de regresión lineal sobre variables estandarizadas ", r_lin)
if r_lin< r_linn: 
     print ( "regresión sobre variables normalizadas  tiene  mejor ajuste")
else: 
    print ( "regresión sobre variables estandarizadas tiene mejor ajuste")


# In[807]:


print('Error cuadrático medio de la regresión lineal  sobre variables normalizadas :', mean_squared_error(yntest, fx_linn))
print('Error cuadrático medio de la regresión lineal sobre variables estandarizadas :', mean_squared_error(y_test, fx_lin))
if  mean_squared_error(y_test, fx_lin)<  mean_squared_error(yntest, fx_linn): 
    print ( "la regresión sobre variables estandarizadas tiene menor error de ajuste ")
else: 
    print ( "la regresión sobre variables normalizadas tiene menor error de ajuste ")


# - **Bosque Aleatorio**

# In[806]:


rftn.fit(Xntrain, yntrain)
fx_rftn= rftn.predict(Xntest)


# In[808]:


r_rftn = rftn.score(Xntest, yntest)
print("r de regresión lineal  sobre variables normalizadas  ", r_rftn)
print("r de regresión lineal sobre variables estandarizadas ", r_rft)
if r_rft< r_rftn: 
     print ( "regresión sobre variables normalizadas  tiene  mejor ajuste")
else: 
    print ( "regresión sobre variables estandarizadas tiene mejor ajuste")


# In[809]:


print('Error cuadrático medio de la regresión lineal  sobre variables normalizadas :', mean_squared_error(yntest, fx_rftn))
print('Error cuadrático medio de la regresión lineal sobre variables estandarizadas :', mean_squared_error(y_test, fx_rft))
if  mean_squared_error(y_test, fx_rft)<  mean_squared_error(yntest, fx_rftn): 
    print ( "la regresión  estándar tiene menor error de ajuste ")
else: 
    print ( "la regresión normalizada tiene menor error de ajuste ")


# - **Nivel 3**. 
# 
# "Haz los cálculos sin contar con DepDelay" 
# 

# In[843]:


df11= df9[["ArrDelay", 'CarrierDelay', 'NASDelay' ,'LateAircraftDelay', 'X10', 'X11']]
Xb= df11.drop(["ArrDelay"], axis=1)

yb= df11["ArrDelay"]

Xtrainb, Xtestb, ytrainb, ytestb = train_test_split( Xb, yb, test_size=0.30, random_state=42, shuffle=True)


# In[844]:


mplb= MLPRegressor(hidden_layer_sizes=(20,20,20), activation="logistic", solver="sgd",max_iter=300 )


# In[845]:


mplb.fit(Xtrainb, ytrainb)
fx_mplb= mplb.predict(Xtestb)


# In[846]:


r_mplb = mplb.score(Xtestb, ytestb)
print("r de regresión lineal sin DepDelay  ", r_mplb)
print("r  de regresión lineal con DepDelay ", r_mpl)
if r_mpl< r_mplb: 
     print ( "red sin DepDelay tiene mejor ajuste")
else: 
    print ( "red con DepDelay tiene mejor ajuste")


# In[847]:


print('Error cuadrático medio de la regresión lineal sin DepDelay :', mean_squared_error(ytestb, fx_mplb))
print('Error cuadrático medio de la regresión lineal con DepDelay :', mean_squared_error(y_test, fx_mpl))
if  mean_squared_error(y_test, fx_mpl)<  mean_squared_error(ytestb, fx_mplb): 
    print ( "la regresión  con DepDelay tiene menor error de ajuste ")
else: 
    print ( "la regresión sin DepDelay tiene menor error de ajuste ")


# En este último caso podemos observar un coeficiente correlación  muy baja, sin embargo un error igual de bajo que con menor
# error de ajust. 
# 
# 

# In[860]:




df13=pd.DataFrame( { "ytestb": ytestb, "fx_mplb":fx_mplb})
df13.head(50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




