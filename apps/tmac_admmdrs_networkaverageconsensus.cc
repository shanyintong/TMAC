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
using namespace std;
//using namespace MyAlgebra;
#include "algebra_namespace_switcher.h"

double objective(Vector theta_, double& avrg, double& weight) ;

int main(int argc, char *argv[]) {
  // Step 0: Define the parameters and input file names  
  Params params;
  //string data_file_name;
  string label_file_name;
  
  // Step 1. Parse the input argument
  //parse_input_argv_mm(&params, argc, argv, label_file_name);
    parse_input_argv_demo(&params, argc, argv);

  // Step 2. Load the data or generate synthetic data, define matained variables
  /* loadMarket(A, data_file_name);
     loadMarket(theta_, label_file_name);
   */
   
  //int N = theta_.size();
  //params.problem_size = N;
  mutex lock_average;
  int N = params.problem_size;
  params.tmac_step_size = 0.01;
  params.max_itrs = 10000;
  params.worker_type = "gs" ;
  Vector theta_(N, 3.);
  for(int i = 0; i < N; i++)
    theta_[i] = i + 1;
    
  Vector x(N, 0.);   // unknown variables, initialized to zero
  double avrg = 0. ;// maintained variables, initialized to zero
  
  // Step 3. Define your three operators based on data and parameters
  double operator_step_size = params.step_size ;
  double weight_gamma = 1.;
  op2_for_network_average_consensus op2(&avrg, operator_step_size, weight_gamma);
  using Second = decltype(op2);
  op1_for_network_average_consensus op1(&theta_, operator_step_size, weight_gamma);
  using First = decltype(op1);
  op3_updating_average_WITHLOCK op3(&avrg, &lock_average, operator_step_size, weight_gamma);
  using Third = decltype(op3);

  // Step 4. Define your operator splitting scheme
  DoglasRachfordSplittingAdmm<First, Second, Third> DRs(&x, op1, op2, op3);

  // Step 6. Call the TMAC function
  double start_time = get_wall_time();  
  TMAC(DRs, params);
  double end_time = get_wall_time();  

  // Step 7. Print results
  print_parameters(params);
  cout << "Computing time is: " << end_time - start_time << endl;
  cout << "Objective value is: " << objective(theta_, avrg, weight_gamma) << endl;
  cout << "---------------------------------" << endl;  
  return 0;
}


double objective(Vector theta_, double& avrg, double& weight) {
  int len = theta_.size();
  double y = -avrg / weight;
  double result = 0.;
  for(int i = 1; i <= len; ++i)
    result += ((theta_)[i] - y) * ((theta_)[i] - y);
  return result;
}