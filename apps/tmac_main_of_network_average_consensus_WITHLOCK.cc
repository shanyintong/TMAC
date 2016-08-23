/**********************************
 * Example 1: solving Network Average Consensus
 * 
 *    min_x sum_{i =1}^N ||x - theta_i||^2
 *
 * Using consensus optimization and solve the dual 
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
#include <mutex>          // std::mutex
using namespace std;
//using namespace MyAlgebra;
#include "algebra_namespace_switcher.h"

double objective(Vector theta_, double& avrg, double& weight) ;

int main(int argc, char *argv[]) {
    
  mutex lock_of_avrg;

  // Step 0: Define the parameters and input file names  
  Params params;
  //string data_file_name;
  string label_file_name;
  
  // Step 1. Parse the input argument
  //parse_input_argv_mm(&params, argc, argv, label_file_name);
    parse_input_argv_demo(&params, argc, argv);

  // Step 2. Load the data or generate synthetic data, define matained variables
  //Vector theta_(5, 3.);   
 // loadMarket(A, data_file_name);
  //loadMarket(theta_, label_file_name);
  //double data_theta[5] = {1., 2., 3., 4., 5.};
  // theta_.assign(data_theta, data_theta + 5);

  //int problem_size = theta_.size();//这个叫什么……
  //params.problem_size = problem_size;
  int problem_size = params.problem_size;
  params.tmac_step_size = 0.01;
  params.max_itrs = 1000;
  params.worker_type = "gs" ;
  //params.async = false;
  Vector theta_(problem_size, 3.);
  for(int i = 0; i < problem_size; i++)
    theta_[i] = i + 1;   
  Vector x(problem_size, 0.);   // unknown variables, initialized to zero
  double avrg = 0. ;// maintained variables, initialized to zero
  // Step 3. Define your three operators based on data and parameters
  double second_operator_step_size = 0.01;
  params.step_size = second_operator_step_size;
  double weight_gamma = 1.;
  op2_for_network_average_consensus<Vector> op2(&theta_, &avrg, second_operator_step_size, weight_gamma);  
  using First = decltype(op2);
  double first_operator_step_size = 0.01;
  params.step_size = first_operator_step_size;

  op1_for_network_average_consensus<Vector> op1(&theta_, first_operator_step_size, weight_gamma);  
  using Second = decltype(op1);

  double third_operator_step_size = 0.01;
  params.step_size = third_operator_step_size;

  op3_for_network_average_consensus<Vector> op3(&theta_, &avrg, &lock_of_avrg, third_operator_step_size, weight_gamma);
  using Third = decltype(op3);

  // Step 4. Define your operator splitting scheme
  DoglasRachfordSplittingAdmm<op1_for_network_average_consensus<Vector>, 
    op2_for_network_average_consensus<Vector>, op3_for_network_average_consensus<Vector> > 
    DRs(&x, op1, op2, op3);

  // Step 6. Call the TMAC function
  double start_time = get_wall_time();  
  TMAC(DRs, params);
  double end_time = get_wall_time();  

  print_parameters(params);

    
  double real_avrg;
    for(int t = 0; t < problem_size; ++t){
     // cout<<"x_" << t << " = "<< x[t] <<' ';
      real_avrg += x[t];
    }
  real_avrg /= problem_size;
  cout<<endl;
  cout<<"avrg = "<< avrg <<endl;
  cout<<"real avrg = "<<real_avrg<<endl;
    
  cout << "Computing time is: " << end_time - start_time << endl;  
  // Step 7. Print results

  cout << "Objective value is: " << objective(theta_, avrg, weight_gamma) << endl;
  cout << "---------------------------------" << endl;  
  return 0;
}


double objective(Vector theta_, double& avrg, double& weight) {
  int len = theta_.size();
  double y = -avrg / weight;
//  double result = 0.;
//  for(int i = 1; i <= len; ++i)
//    result += ((theta_)[i] - y) * ((theta_)[i] - y);
  return y;
}