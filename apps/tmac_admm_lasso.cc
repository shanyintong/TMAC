/**********************************
 * Example 2: lasso by ADMM-DRS, which n = 1
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

double objective(Vector* A, Vector* b, double y, double lambda_);
void get_result(double* avrg, double* y, double lambda_, double weight, int N);
void set_A(Vector* A, int N);
void set_b(Vector* b, int N);
void cal_inverse(Vector* A, Vector* inverse_matrices, double weight);
void cal_Atb(Vector* A, Vector* b, Vector* Atbs);

int main(int argc, char *argv[]) {
    mutex average_lock;
    // Step 1. Parse the input argument
    Params params;
    string label_file_name;
    parse_input_argv_demo(&params, argc, argv);
    int problem_size = params.problem_size;
    params.tmac_step_size = 0.5;
    params.max_itrs = 100000;
    params.worker_type = "cyclic" ;
    double weight_gamma = 1.;

    // Step 2. Load the data or generate synthetic data, define matained variables
    int N = problem_size;
    double lambda_ = 3.;
    double avrg = 0.07213;
    Vector A(N, 0.);
    Vector b(N, 0.);
    Vector x(N, 0.07213);
    Vector inverse_matrices(N, 0.);
    Vector Atbs(N, 0.);
    set_A(&A, N);
    set_b(&b, N);
    cal_inverse(&A, &inverse_matrices, weight_gamma);
    cal_Atb(&A, &b, &Atbs);
 /*
    print(A);
    print(b);
    print(inverse_matrices);
    print(Atbs);
*/
    double y = 0;
    
    // Step 3. Define your three operators based on data and parameters
    double operator_step_size = 0.5;
    params.step_size = operator_step_size;

    op1_for_ADMMI_Lasso<Vector> op1(&inverse_matrices, &Atbs, operator_step_size, weight_gamma);
    using First = decltype(op1);
    op2_for_ADMMI_Lasso<Vector> op2(&A, &b, &avrg, lambda_, operator_step_size, weight_gamma);
    using Second = decltype(op2);
    op3_updating_average_WITHLOCK<Vector> op3(&avrg, &average_lock, operator_step_size, weight_gamma);
    using Third = decltype(op3);
    
    // Step 4. Define your operator splitting scheme
    DoglasRachfordSplittingAdmm<First, Second, Third> DRs(&x, op1, op2, op3);

    // Step 6. Call the TMAC function
    double start_time = get_wall_time();
    TMAC(DRs, params);
    double end_time = get_wall_time();
 
    print_parameters(params);
    cout << "Computing time is: " << end_time - start_time << endl;
    // Step 7. Print results
    get_result(&avrg, &y, lambda_, weight_gamma, N);
    cout << "avrg = "<< avrg <<endl;
    cout << "y = "<< y <<endl;
    cout << "Objective value is: " << objective(&A, &b, y, lambda_) << endl;
    cout << "---------------------------------" << endl;
    
    
    return 0;
    
}

double objective(Vector* A, Vector* b, double y, double lambda_){
    double result = 0.;
    for(int i = 0; i < A->size(); ++i)
        result += ((*A)[i] * y - (*b)[i]) * ((*A)[i] * y - (*b)[i]) / 2;
        result += lambda_ * abs(y);
    return result;
}
void get_result(double* avrg, double* y, double lambda_, double weight, int N){
    prox_l1 thres(lambda_ / N);
    double S = thres(-(*avrg));
    *y = S / weight;
}
void set_A(Vector* A, int N){
    for(int i = 0; i < N; ++i)
        (*A)[i] = 1 + i;
}
void set_b(Vector* b, int N){
    for(int i = 0; i < N; ++i)
        (*b)[i] = N - i;
   // (*b)[0] += 1;
}
void cal_inverse(Vector* A, Vector* inverse_matrices, double weight){
    for(int i = 0; i < A->size(); ++i)
        (*inverse_matrices)[i] = 1. / ((*A)[i] * (*A)[i] + weight);
}
void cal_Atb(Vector* A, Vector* b, Vector* Atbs){
    for(int i = 0; i < A->size(); ++i)
        (*Atbs)[i] = (*A)[i] * (*b)[i];
}
