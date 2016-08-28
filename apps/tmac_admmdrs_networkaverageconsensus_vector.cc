/**********************************
 * Example 1: solving Network Average Consensus, which the basic units are vectors
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
#include <mutex>          // std::mutex


void get_result(Vector* avrg, Vector* results, double weight, int unit_size);
mutex lock_of_avrg;

int main(int argc, char *argv[]) {
    
  int unit_size = 500;
  Vector results;
  results.resize(unit_size);
   
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
  params.max_itrs = 10000;
  params.worker_type = "gs" ;
  double weight_gamma = 1.;
    
  vector<Vector> theta_;
  for(int i = 0; i < problem_size; ++i){
    Vector theta_i(unit_size, (double)i+1);
    for(int j = 0; j < unit_size; ++j)
        theta_i[j] += j;
    theta_.push_back(theta_i);
  }

  vector<Vector> x;// unknown variables, initialized to zero
  for(int i = 0; i < problem_size; ++i){
    Vector x_i(unit_size, 0.);
    x.push_back(x_i);
  }

  Vector avrg(unit_size, 0.);// maintained variables, initialized to zero

  // Step 3. Define your three operators based on data and parameters
  double second_operator_step_size = 0.01;
  params.step_size = second_operator_step_size;

  op2_for_network_average_consensus_vector<vector<Vector>> op2(&theta_, &avrg, second_operator_step_size, weight_gamma);
  using Second = decltype(op2);
  double first_operator_step_size = 0.01;
  params.step_size = first_operator_step_size;
  
  op1_for_network_average_consensus_vector<vector<Vector>> op1(&theta_, first_operator_step_size, weight_gamma);
  using First = decltype(op1);

  double third_operator_step_size = 0.01;
  params.step_size = third_operator_step_size;

  op3_for_network_average_consensus_vector_WITHLOCK<vector<Vector>> op3(&theta_, &avrg, &lock_of_avrg, third_operator_step_size, weight_gamma);
  using Third = decltype(op3);
   
  // Step 4. Define your operator splitting scheme
  DoglasRachfordSplittingAdmm_vector<First, Second, Third> DRs(&x, op1, op2, op3);

  // Step 6. Call the TMAC function
  double start_time = get_wall_time();
  TMAC(DRs, params);
  double end_time = get_wall_time();  

  print_parameters(params);

  cout << "Computing time is: " << end_time - start_time << endl;  
  // Step 7. Print results
  get_result(&avrg, &results, weight_gamma, unit_size);
  //cout << "Objective value is: " << objective(theta_, avrg, weight_gamma) << endl;
  print(results);
  cout << "---------------------------------" << endl;  
  return 0;
}


void get_result(Vector* avrg, Vector* results, double weight, int unit_size){
  //int len = theta_.size();
  copy(*avrg, *results, 0, unit_size);
  scale(*results, -1./(double)weight);
}