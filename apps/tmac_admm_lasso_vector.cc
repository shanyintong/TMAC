//  tmac_admm_lasso_vector.cpp
//  Created by LuLingxi on 16/8/26.
/**********************************
 * Example 2: lasso by ADMM-DRS, which A_i \in 1 x n
 *
 *   min sum_{i =1}^N (1/2||A_iz_i - b_i||_2^2) + ||y||_1
 *     s.t. z_i = y
 *
 * Using ADMM I and solve the dual
 *         problem by ADMM - DRS form
 *
 *********************************/

#include <iostream>
#include "matrices.h"
#include "algebra.h"
#include "operators.h"
#include "parameters.h"
#include "splitting_schemes.h"
#include "tmac.h"
#include "util.h"
#include <thread>
using namespace std;
//using namespace MyAlgebra;
#include "algebra_namespace_switcher.h"
#include <mutex>          // std::mutex

// declare the needed LAPACK functions
// extern "C": let C++ code call a C-style library
// DGETRF computes an LU factorization of a matrix A using partial pivoting with row interchanges
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, int* ipiv, int* info);
// DGETRI computes the inversion of a matrix using the LU factorization computed by DGETRF
extern "C" void dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK, int *LWORK, int *INFO);

double objective(vector<Vector>* A, Vector* b, Vector* y, double lambda_, int N, int n_unit);
void get_result(Vector* avrg, Vector* y, double lambda_, double weight, int N, int n_unit);
void set_A(vector<Vector>* A, int N, int n_unit);
void set_b(Vector* b, int N);
void init_x_avrg(Vector* avrg, vector<Vector>* x, int N, int n_unit);
void cal_inverse(vector<Vector>* A, vector<Matrix>* inverse_matrices, double weight, int N, int n_unit);
void cal_Atb(vector<Vector>* A, Vector* b, vector<Vector>* Atbs, int N, int n_unit);
void get_inverse(Matrix* original_, int n_unit, int* info);


int main(int argc, char *argv[]) {
    mutex average_lock;
    // Step 1. Parse the input argument
    Params params;
    string label_file_name;
    parse_input_argv_demo(&params, argc, argv);
    int problem_size = params.problem_size;
    params.tmac_step_size = 0.5;
    params.max_itrs = 10000;
    params.worker_type = "cyclic" ;
    double weight_gamma = 1.;

    // Step 2. Load the data or generate synthetic data, define matained variables
    int N = problem_size;
    int n_unit = 50;
    double lambda_ = 9.;
    Vector avrg;
    vector<Vector> A;
    Vector b;
    vector<Matrix> inverse_matrices;
    vector<Vector> Atbs;
    vector<Vector> x;

    set_A(&A, N, n_unit);
    set_b(&b, N);
    init_x_avrg(&avrg, &x, N, n_unit);
    
    double first_time = get_wall_time();
    cal_inverse(&A, &inverse_matrices, weight_gamma, N, n_unit);
    cal_Atb(&A, &b, &Atbs, N, n_unit);
   
    /*
    for(int i = 0; i < N; ++i)
        print(A[i]);
    print(b);
    for(int i = 0; i < N; ++i)
        print(inverse_matrices[i]);
    for(int i = 0; i < N; ++i)
        print(Atbs[i]);
    */
     
    Vector y(n_unit);
    
    // Step 3. Define your three operators based on data and parameters
    double operator_step_size = 0.5;
    params.step_size = operator_step_size;
    
    op1_for_ADMMI_lasso_vector op1(&inverse_matrices, &Atbs, operator_step_size, weight_gamma);
    using First = decltype(op1);
    op2_for_ADMMI_lasso_vector op2(&avrg, lambda_, N, operator_step_size, weight_gamma);
    using Second = decltype(op2);
    op3_updating_average_vector_WITHLOCK<Vector> op3(&avrg, &average_lock, N, operator_step_size, weight_gamma);
    using Third = decltype(op3);
    
    // Step 4. Define your operator splitting scheme
    DoglasRachfordSplittingAdmm_vector<First, Second, Third> DRs(&x, op1, op2, op3);

    
    // Step 6. Call the TMAC function
    double start_time = get_wall_time();
    TMAC(DRs, params);
    double end_time = get_wall_time();

    print_parameters(params);
    
    cout << "Preparation time is: " << start_time - first_time << endl;
    cout << "Computing time is: " << end_time - start_time << endl;
    // Step 7. Print results
    get_result(&avrg, &y, lambda_, weight_gamma, N, n_unit);
    print(avrg);
    print(y);
    cout << "Objective value is: " << objective(&A, &b, &y, lambda_, N, n_unit) << endl;
    cout << "---------------------------------" << endl;
    
    return 0;
}
double objective(vector<Vector>* A, Vector* b, Vector* y, double lambda_, int N, int n_unit){
    double result = 0;
    for(int i = 0; i < N; ++i){
        double temp = 0;
        for(int j = 0; j < n_unit; ++j)
            temp += (*A)[i][j] * (*y)[j];
        result += (temp - (*b)[i]) * (temp - (*b)[i]) / 2;
    }
    for(int i = 0; i < n_unit; ++i)
        result += lambda_ * abs((*y)[i]);
    return result;
}
void get_result(Vector* avrg, Vector* y, double lambda_, double weight, int N, int n_unit){
    y->resize(avrg->size());
    prox_l1 thres(lambda_ / N);
    Vector temp;
    temp.resize(avrg->size());
    copy(*avrg, temp, 0, avrg->size());
    scale(temp, -1.);
    thres(&temp, y);
    scale(*y, 1./weight);
}


void set_A(vector<Vector>* A, int N, int n_unit){
    for(int i = 0; i < N; ++i){
        Vector A_i;
        for(int j = 0; j < n_unit; ++j){
            if(i == j)
                A_i.push_back(j+1);
            else
                A_i.push_back(0);
        }
        A->push_back(A_i);
    }
    /*
    Vector A_1;
    A_1.push_back(2.);
    A_1.push_back(3.);
    A->push_back(A_1);
    Vector A_2;
    A_2.push_back(6.);
    A_2.push_back(-5.);
    A->push_back(A_2);
    */
}

void set_b(Vector* b, int N){
    for(int i = 0; i < N; ++i)
        b->push_back(N - i);
    /*
    b->push_back(1.);
    b->push_back(-7.);
     */
}

void get_inverse(Matrix* a, int n_unit, int* info){
    int dim = n_unit;        // 2x2 matrix
    int LDA = dim;
    int lwork = dim*dim;
    // array of pivot indices; for 1 <= i <= min(M,N), row i of the matrix was interchanged with row IPIV(i).
    int ipiv[dim];
    // LU factorization, "ipiv" gets pivot info, "a" gets L and U without the unit diagonals of L
    dgetrf_(&dim, &dim, &*(*a).begin(), &LDA, ipiv, info);

    double *b1 = new double[dim]();
    // Compute inversion
    dgetri_(&dim, &*(*a).begin(), &LDA, ipiv, b1, &lwork, info);
   // delete [] b1;
}

void cal_inverse(vector<Vector>* A, vector<Matrix>* inverse_matrices, double weight, int N, int n_unit){
    for(int i = 0; i < N; ++i){
        Matrix AAt(n_unit, n_unit);
        int info;
        for(int p = 0; p < n_unit; ++p)
            for(int q = 0; q < n_unit; ++q)
                AAt(p, q) = (*A)[i][p] * (*A)[i][q];
        for(int p = 0; p < n_unit; ++p)
            AAt(p, p) += weight;
        get_inverse(&AAt, n_unit, &info);
        inverse_matrices->push_back(AAt);
    }
}
void cal_Atb(vector<Vector>* A, Vector* b, vector<Vector>* Atbs, int N, int n_unit){
    for(int i = 0; i < N; ++i){
        double scale_ = (*b)[i];
        Vector temp;
        temp.resize(n_unit);
        copy((*A)[i], temp, 0, n_unit);
        scale(temp, scale_);
        Atbs->push_back(temp);
    }
}
void init_x_avrg(Vector* avrg, vector<Vector>* x, int N, int n_unit){
    for(int i = 0; i < N; ++i){
        Vector x_i(n_unit, 0.);
        x->push_back(x_i);
    }
    for(int i = 0; i < n_unit; ++i)
        avrg->push_back(0.);
}
