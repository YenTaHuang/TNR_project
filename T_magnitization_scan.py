from mainTNR import *
import matplotlib.pyplot as plt

chiM = 8
chiS = 4
chiU = 4
chiH = 6
chiV = 6

numlevels = 40 # number of coarse-grainings

time_start =time.asctime()

path1 = "./T_magnitization_scan_"+str(time_start)+".txt"
file1 = open(path1,'w')
#relTemp_list = [1.1 + 0.05*n for n in range(12)]
relTemp_list = [0.4 + 0.05*n for n in range(10)] + [0.9 + 0.001*n for n in range(200)] + [1.1 + 0.05*n for n in range(12)]

#relTemp_list = [0.5,0.6,0.7,0.8,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.2,1.3,1.4,1.5,1.6]
#relTemp_list = [0.6 + 0.1*n for n in range(3)]

O_dtol = 1e-10
O_disiter = 2000
O_miniter = 200
O_dispon = False
O_convtol = 0.01
O_disweight = 0

T_magnitization_list = []

for relTemp in relTemp_list:
    print("################ relTemp = ",relTemp," starts #################")
    magnitization = mainTNR(relTemp,allchi,numlevels,O_dtol,O_disiter,O_miniter,O_dispon,O_convtol,O_disweight)
    T_magnitization_list.append([relTemp,magnitization])

print("T_magnitization_list: ",T_magnitization_list)

file1.write(str(T_magnitization_list).replace("e","*^").replace("[","{").replace("]","}"))
file1.close()

data = np.array(T_magnitization_list)
x, y= data.T
plt.scatter(x, np.abs(y))

plt.savefig('T_magnitization_list_'+str(time_start)+'.pdf')
plt.show()
