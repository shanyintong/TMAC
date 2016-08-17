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

double objective(Vector& theta_, Vector& x) ;

int main(int argc, char *argv[]) {

  // Step 0: Define the parameters and input file names  
  Params params;
  //string data_file_name;
  string label_file_name;
  
  // Step 1. Parse the input argument
  parse_input_argv_mm(&params, argc, argv, label_file_name);

  // Step 2. Load the data or generate synthetic data, define matained variables
  Vector theta_;   
 // loadMarket(A, data_file_name);
  loadMarket(theta_, label_file_name);
  int problem_size = theta_.size();
  params.problem_size = problem_size;
  params.tmac_step_size = 0.1;
 // int sample_size = A.cols();
  Vector x(problem_size, 0.);   // unknown variables, initialized to zero
  double avrg = 0. // maintained variables, initialized to zero
  // Step 3. Define your three operators based on data and parameters
  double second_operator_step_size = 0.0005;
  params.step_size = second_operator_step_size;
  double weight_gamma = 1.;
  op2_for_network_average_consensus<Vector> op2(&theta_, &avrg, second_operator_step_size, weight_gamma);  
  using First = decltype(op1);

  double first_operator_step_size = 0.0005;
  params.step_size = first_operator_step_size;

  op1_for_network_average_consensus<Vector> op1(&theta_, first_operator_step_size, weight_gamma);  
  using Second = decltype(op1);

  double third_operator_step_size = 0.0005;
  params.step_size = third_operator_step_size;

  op3_for_network_average_consensus<Vector> op3(&theta_, &avrg, third_operator_step_size, weight_gamma);  
  using Third = decltype(op3);

  // Step 4. Define your operator splitting scheme
  DoglasRachfordSplittingAdmm<op1_for_network_average_consensus<Vector>, 
    op2_for_network_average_consensus<Vector>, op3_for_network_average_consensus<Vector> > 
    gd(&x, &avrg, op1, op2, op3);

  // Step 6. Call the TMAC function
  double start_time = get_wall_time();  
  TMAC(gd, params);
  double end_time = get_wall_time();  

  print_parameters(params);
  
  cout << "Computing time is: " << end_time - start_time << endl;  
  // Step 7. Print results

  cout << "Objective value is: " << objective(theta_, avrg) << endl;
  cout << "---------------------------------" << endl;  
  return 0;
}


double objective(Vector& theta_, double& avrg) {
  int len = theta_.size();
  double y = -avrg / weight_gamma;
  double result = 0.;
  for(int i = 1; i <= len; ++i)
    result += ((*theta_)[i] - y) * ((*theta_)[i] - y);
  return result;
}