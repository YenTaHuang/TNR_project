#!/usr/bin/env python3

from TNR_lib import *

def main():

    chi_p = 8

    NRG = 40

    N1 = 50000
    N2 = 50000
    N3 = 50000

    delta = 0.1
    h = 0.00
    beta_list = [0.4]

    path1 = "./beta_magnitization_list_chi"+str(chi_p)+"_NRG"+str(NRG)+".txt"
    path2 = "./beta_free_energy_list_chi"+str(chi_p)+"_NRG"+str(NRG)+".txt"
    path3 = "./beta_nRG_svd_A_list_chi"+str(chi_p)+"_NRG"+str(NRG)+".txt"
    file1 = open(path1,'w')
    file2 = open(path2,'w')
    file3 = open(path3,'w')

    file1.write("{")
    file2.write("{")
    file3.write("{")

    for beta in beta_list:

        time1 = time.time()
        print("########## beta = ",beta," starts #########")

        A = initialize_A_with_h(beta,h)
        M = initialize_M_with_h(beta,h)

        print("eigenvalue of new A: ",eig_A(A))
        print("svd eigenvalue of new A: ",svd_A(A))

        A_old,M_old,B_old,D_old,A_prime_old,error,Z_old = update_AM(A,M,chi_p,N1,N2,N3)
        print("magnitization of pre-step: ",expectation_value(A_old,M_old))

        eig_A_result_old = eig_A(A_old)
        svd_A_result_old = svd_A(A_old)
        A_matrix = np.reshape(A_old,(A_old.shape[0]**2,A_old.shape[0]**2))
        print("||A_matrix-ct(A_matrx)|| = ",magnitude(A_matrix-ct(A_matrix)))
        print("eigenvalues of new A: ",eig_A_result_old)
        print("svd eigenvalues of new A: ",svd_A_result_old)

        print("Z_initial = ",Z_old)


        magnitization_list = []
        svd_A_list = []
        nRG = 0
        check = True
        while check:
            A_new,M_new,B_new,D_new,A_prime_new,error,Z_new = update_AM(A_old,M_old,chi_p,N1,N2,N3,B_old,D_old,A_prime_old,delta,Z_old)
            magnitization = expectation_value(A_new,M_new)
            magnitization_list.append(magnitization)
            eig_A_result_new = eig_A(A_new)
            svd_A_result_new = svd_A(A_new)
            svd_A_list.append([nRG,svd_A_result_new.tolist()])
            print("magnitization of ",nRG,"th step: ",magnitization)
            eig_A_diff = magnitude(eig_A_result_new-eig_A_result_old)
            svd_A_diff = magnitude(svd_A_result_new-svd_A_result_old)
            A_matrix = np.reshape(A_new,(A_new.shape[0]**2,A_new.shape[0]**2))
            print("||A_matrix-ct(A_matrx)|| = ",magnitude(A_matrix-ct(A_matrix)))
            print("eigenvalues of new A: ",eig_A_result_new)
            print("svd eigenvalues of new A: ",svd_A_result_new)
            print("eig_A_diff: ",eig_A_diff)
            print("svd_A_diff: ",svd_A_diff)
            epsilon_A = magnitude(A_new-A_old)
            print("||A_new-A_old|| = ",epsilon_A)
            print("Z_new/Z_old = ",Z_new/Z_old)

            A_old = A_new
            M_old = M_new

            B_old = B_new
            D_old = D_new
            A_prime_old = A_prime_new
            Z_old = Z_new
            eig_A_result_old = eig_A_result_new
            svd_A_result_old = svd_A_result_new

            if (nRG > NRG or epsilon_A<5e-3) and (nRG>7 or magnitization>0.1):
                check = False

            nRG += 1



        free_energy = -1/beta*np.log(Z_old)
        print("For beta = ",beta,", free energy: ",free_energy)
        file1.write(('['+str(beta)+','+str(magnitization_list)+'],').replace("e","*^").replace("[","{").replace("]","}"))

        file2.write(('['+str(beta)+','+str(free_energy)+'],').replace("e","*^").replace("[","{").replace("]","}"))
        file3.write(('['+str(beta)+','+str(svd_A_list)+'],').replace("e","*^").replace("[","{").replace("]","}"))


        time2 = time.time()
        print("time spent for this beta: ",time2-time1)
        print("#################################")
        print("#################################")



    file1.write("}")
    file2.write("}")
    file3.write("}")
    file1.close()
    file2.close()
    file3.close()

if __name__ == "__main__":
    main()
