import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import pandas as pd
import networkx as nx
import os
import random
import itertools

#-----------------------------------------------------
#               FUNCIONES PARA EL TP3:
#-----------------------------------------------------
def rewiring(G):
    #Funcion de grafo G que toma un grafo y realiza un recableado
    #manteniendo el grado de cada nodo.
    #Estrategia que vimos en clase de Cherno de redes random:
    #1) Realizamos una copia del grafo G sin enlaces.
    #2) Vamos tomando pares de nodos al azar y creamos enlaces manteniendo el grado de cada nodo hasta agotar el cupo.
    #Este proceso nos va a dejar con tres tipos de enlaces entre nodos: e. simples, e. autoloops y e. multienlace. Estos dos últimos son enlaces problemáticos
    #y hay que eliminarlos.
    #3)Eliminamos los autoloops.
    #4)Eliminamos los multienlaces
    #5)Corroboramos que no queden ni autoloops ni multienlaces.
    #6)Ultimo chequeo para corroborar que no haya cambiado el grado de cada nodo.
    
    nodos=list(G.nodes)
    enlaces=list(G.edges)
    grados_dict = dict(G.degree())
    k_nodo= list(grados_dict.values()) # lista de grados de cada nodo en nodos(ordenados)
    #print('grado G')
    #print('enlaces originales en G sin autoloops: {}'.format(len(enlaces))) #hemos removido autoloops en el programa principal

    #Ahora nos quedamos con nodos de k distinto de 0 de esa forma mantengo el k=0 para los nodos aislados:
    index_nonzerok=[i for i in range(0,len(k_nodo)) if k_nodo[i]!=0] #buscamos los lugares de k_control donde no hayan quedado zeros
    k_nodo=[k_nodo[i] for i in range(0,len(k_nodo)) if k_nodo[i]!=0]
    k_nodo_antes=k_nodo
    nodos=[nodos[index_nonzerok[i]] for i in range(0,len(index_nonzerok))]
    enlaces=list(G.edges)
    
    #1) Creo un multigraph D que acepte multiedges
    D = nx.MultiGraph()

    #Agrego nodos:
    D.add_nodes_from(nodos) #Nota D solo va a tener los nodos de G que se hallan conectdos, no posee los nodos de G que se encuentran aislados
    
    #Inicializo k_control y nodos_new:
    k_control=np.array(k_nodo) #cuando creo un enlace entre nodoi y nodoj se le restara un 1 a los lugares i y j de k_control
    nodos_new=nodos

    #2)Agregamos enlaces de forma aleatoria al grafo D, manteniendo controlado que el cupo de cada nodo no puede exceder su grado.
    while(len(nodos_new)>0): 
        #Elijo uno random de pairs
        pair= np.random.choice(nodos_new,(1,2),replace=True)[0] #al poner replace True permito sacar dos numeros iguales.(eso va a crear un autoloop)
                   
        #Actualizamos variable de control: k_control
        if pair[0] == pair[1]:
            if k_control[nodos.index(pair[0])]>1:
                k_control[nodos.index(pair[0])]=k_control[nodos.index(pair[0])]-2 #solo actualizamos ese y le restamos un 2 ya que se creo un autoloop
                #creamos el autoloop
                D.add_edge(pair[0], pair[1])
                #no es posible crear el autoloop si habia un 1 en k_nodos, como mínimo necesito tener un 2 o mayor en el vector de grado de ese nodo.
        else:
            #creamos el enlace el cual no es un autoloop
            D.add_edge(pair[0], pair[1])
            k_control[nodos.index(pair[0])]=k_control[nodos.index(pair[0])]-1 #actualizamos k_control en la pos i
            k_control[nodos.index(pair[1])]=k_control[nodos.index(pair[1])]-1 #actualizamos k_control en la pos j

    
        #Actualizamos variable de control: nodos_new
        if k_control[nodos.index(pair[0])]==0 or k_control[nodos.index(pair[1])]==0: #solo actualizo k_control cuando alguno de los valores llega a cero
            index_nonzerok=[i for i in range(0,len(k_control)) if k_control[i]>0] #buscamos los lugares de k_control donde hayan elementos dinstintos a cero
            index_equalzero=[i for i in range(0,len(k_control)) if k_control[i]==0]#buscamos los lugares de k_control donde hayan elementos igual a cero
            nodos_new=[nodos[index_nonzerok[i]] for i in range(0,len(index_nonzerok))] #actualizamos la lista de nodos asi no volvemos a tomar nodos que ya recableamos por completo o sea aquellos que alcanzaron k_control[i]=0   
      

    #print('grafico D')
    enlaces=list(D.edges())
    #Enlaces problemáticos:
    #Selfloops:
    '''
    print('autoloops inicial: {}'.format(len(list(D.nodes_with_selfloops()))))
    autoloops=list(D.nodes_with_selfloops())
    '''
    #print('autoloops inicial: {}'.format(len(list(D.selfloop_edges()))))
    autoloops=list(D.selfloop_edges())
    #print(autoloops)
    
    #Multiples y Simples:
    enlaces_multiples=[]
    enlaces_simples=[]
    
    for i in range(0,len(nodos)):
        for j in range(i+1,len(nodos)):
            if(D.number_of_edges(nodos[i],nodos[j]))>1:
                for k in range(0,D.number_of_edges(nodos[i],nodos[j])-1):#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo menos 1(porque si hay 3 enlaces entre dos nodos solo hay que sacar 2 de ellos )                    
                    enlaces_multiples.append([nodos[i],nodos[j]]) #agrego multiplicidad -1 de enlaces muliples
                enlaces_simples.append([nodos[i],nodos[j]]) #agrego uno simple
            elif (D.number_of_edges(nodos[i],nodos[j]))==1:
                enlaces_simples.append([nodos[i],nodos[j]])
    
    #print('multiples inicial: {}'.format(len(enlaces_multiples)))
    #print('simples inicial: {}'.format(len(enlaces_simples)))
    
    #Comparamos grados en esta etapa intermedia si queremos:
    grados_dict = dict(D.degree())
    k_nodo_despues= list(grados_dict.values())

    #Hasta acá el programa lo que hizo fue reconectar las puntas conservando el constraint de los grado de los nodos.
    #El problema de esto es que aparecieron autoloops en un mismo nodo y multienlaces entre nodos distintos.
    #Estos enlaces los vamos a llamar enlaces problemáticos.

    #Por ultimo hay que eliminar estos enlaces que son problemáticos:
    numero_autoloops=len(autoloops)
    numero_enlaces_multiples=len(enlaces_multiples)

    #3) Eliminemos autoloops primero:
    #print('Recableando autoloops...')
    while(numero_autoloops >0):
        for al in autoloops:
            idx = np.random.choice(len(enlaces_simples),1)[0] #elijo un enlace dentro de los simples o sea no problematicos
            enlace_elegido=enlaces_simples[idx]
            if (enlace_elegido[0]!=al[0]) & (enlace_elegido[1]!=al[0]): #acepto ese enlace al azar si ninguno es el nodo donde esta el autoloop. esto evita crear un nuevo autoloop en el ismo nodo.
                #Hago el swap:
                #Creo dos nuevos
                D.add_edge(al[0],enlace_elegido[0])
                D.add_edge(al[0],enlace_elegido[1])
                #Elimino dos
                D.remove_edge(al[0],al[0])
                D.remove_edge(enlace_elegido[0],enlace_elegido[1])
                #Recalculamos autoloops y numero_autoloops en cada paso:
                autoloops=list(D.selfloop_edges())
                numero_autoloops=len(autoloops)
                #Tengo que actualizar enlaces simples:
                enlaces_simples.remove([enlace_elegido[0],enlace_elegido[1]])
                enlaces_simples.append([al[0],enlace_elegido[0]])
                enlaces_simples.append([al[0],enlace_elegido[1]])
    #print('autoloops intermedio: {}'.format(len(list(D.nodes_with_selfloops()))))

    
    #Actualizamos los enlaces multiples:
    #Multiples y Simples:
    enlaces_multiples=[]
    enlaces_simples=[]
  
    for i in range(0,len(nodos)):
        for j in range(i+1,len(nodos)):
            if(D.number_of_edges(nodos[i],nodos[j]))>1:
                for k in range(0,D.number_of_edges(nodos[i],nodos[j])-1):#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo menos 1(porque si hay 3 enlaces entre dos nodos solo hay que sacar 2 de ellos ).
                    enlaces_multiples.append([nodos[i],nodos[j]])#agrego multiplicidad -1 de enlaces muliples
                enlaces_simples.append([nodos[i],nodos[j]]) #agrego uno simple
            elif (D.number_of_edges(nodos[i],nodos[j]))==1:
                enlaces_simples.append([nodos[i],nodos[j]])
    #print('multiples intermedio: {}'.format(len(enlaces_multiples)))
    #print('simples intermedio: {}'.format(len(enlaces_simples)))

    
    #4) Eliminamos los enlaces multiples:
    numero_enlaces_multiples=len(enlaces_multiples)
    #print('Recableando multiples...')
    while(numero_enlaces_multiples >0):
        for em in enlaces_multiples:
            idx = np.random.choice(len(enlaces_simples),1)[0] #elijo un enlace dentro de los simples o sea no problematicos
            enlace_elegido=enlaces_simples[idx]
            loscuatronodos=[em[0],em[1],enlace_elegido[0],enlace_elegido[1]]
            A = nx.to_pandas_adjacency(D)
            a1=A[em[0]][enlace_elegido[0]]
            a2=A[em[0]][enlace_elegido[1]]
            a3=A[em[1]][enlace_elegido[0]]
            a4=A[em[1]][enlace_elegido[1]]
            adjacencynumber=a1+a2+a3+a4
            #A continuación solo recableamos si los 4 nodos son distintos sino no, porque puedo vovler a crear un autoloop y  ademas...
            #solo recableamos si son enlaces adyacentes sino no, esto evita que se vuelvan a formar mutienlaces.
            controlnumber=adjacencynumber + len(np.unique(loscuatronodos))
            if (controlnumber==4):
                #Hago el swap:
                #Creo dos nuevos
                D.add_edge(em[0],enlace_elegido[0])
                D.add_edge(em[1],enlace_elegido[1])
                #Elimino dos
                D.remove_edge(em[0],em[1])
                D.remove_edge(enlace_elegido[0],enlace_elegido[1])
                #Tengo que actualizar enlaces simples:
                enlaces_simples.remove([enlace_elegido[0],enlace_elegido[1]])
                enlaces_simples.append([em[0],enlace_elegido[0]])
                enlaces_simples.append([em[1],enlace_elegido[1]])
                #Tengo que actualizar enlaces_multiples
                enlaces_multiples.remove([em[0],em[1]])
                numero_enlaces_multiples=len(enlaces_multiples)

    #5)Nos fijamos que no hallan quedado autoloops:
    autoloops=list(D.nodes_with_selfloops())
    #print('autoloops final: {}'.format(len(list(D.nodes_with_selfloops()))))
    
    #Por ultimo me fijo los multiples al final:(deberia ser cero)
    enlaces_multiples=[]
    enlaces_simples=[]
    for i in range(0,len(nodos)):
        for j in range(i+1,len(nodos)):
            if(D.number_of_edges(nodos[i],nodos[j]))>1:
                for k in range(0,D.number_of_edges(nodos[i],nodos[j])-1):
                    enlaces_multiples.append([nodos[i],nodos[j]])#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo.
                enlaces_simples.append([nodos[i],nodos[j]]) #agrego uno simple
            elif (D.number_of_edges(nodos[i],nodos[j]))==1:
                enlaces_simples.append([nodos[i],nodos[j]])
    #print('multiples final: {}'.format(len(enlaces_multiples)))
    #print('simples final: {}'.format(len(enlaces_simples)))   

    #6) Chequeo final para ver que se mantuvo el grado k de los nodos:
    grados_dict = dict(D.degree())
    k_nodo_despues= list(grados_dict.values())
    diferencia=np.array(k_nodo_despues)-np.array(k_nodo_antes)
    #if (len(np.where(diferencia!=0)[0])==0):
        #print('Rewiring exitoso')
    
    return(D)
#-----------------------------------------------------------------------------------
def rewiring_easy(G):
    #Funcion de grafo G que toma un grafo y realiza un recableado
    #manteniendo el grado de cada nodo.

    numero_enlaces=G.number_of_edges()
    
    #Realizamos un numero de swaps del orden de los nodos de la red
    for i in range(0,int(numero_enlaces)):
        nx.double_edge_swap(G, nswap=1)
 
    return(G)
