#ifndef TMAC_INCLUDE_SPLITTING_SCHEMES_H
#define TMAC_INCLUDE_SPLITTING_SCHEMES_H

#include "operators.h"
#include "parameters.h"
#include <typeinfo>
#include "stdlib.h"
#include <iostream>
/****************************************************************
 * Header file for defining the splitting schemes. Each splitting
 * scheme is a functor. The functor contains the pointers to the
 * relevant data, related parameters, maintained variables and
 * unknown variable. It defines the following two functions.
 *
 * void operator(int index) {
 *     // update the unknown variable x at index
 *   // update the maintained variables
 * }
 *
 * void update_params(Params* params) {
 *   // update the operator related parameters
 *   // update the relaxation parameter    
 * }
 *
 * The constructure should also be defined
 *
 *   OperatorSplitting(argument list) {
 *     // initialize the member variables with the input arguments
 *   }
 ***************************************************************/

class SchemeInterface {
public:
  //update internal scheme parameters
  virtual void update_params(Params* params) = 0;
  //compute and apply coordinate update, return S_{index}
  virtual double operator() (int index) =0;
  //compute and store S_{index} in variable S_i
  virtual void operator() (int index, double& S_i) =0;
  //apply block of S stored in s to solution vector
  virtual void update(Vector& s, int range_start, int num_cords) = 0;
  //apply coordinate of S stored in s to solution vector
  virtual void update(double s, int idx ) = 0;
  //update rank worth of cache_vars based on num_threads
  virtual void update_cache_vars(int rank, int num_threads) = 0;
};


// PPA:
template <typename Backward>
class ProximalPointAlgorithm : public SchemeInterface {
public: 
  Backward prox;
  Vector* x;
  double relaxation_step_size;
  ProximalPointAlgorithm(Vector* xx, Backward p, double s) {
    x = xx;
    prox = p;
    prox.update_step_size(s);
  }
  
  void update_params(Params* params) {
    prox.update_step_size(params->get_step_size());
    relaxation_step_size = params->get_tmac_step_size();
  }
  
  double operator() (int index) {
    // Step 1: read the old x[index]
    double old_x_at_idx = (*x)[index];
    // Step 2: local calculation
    double S_i = old_x_at_idx - prox(x, index);
    // Step 3: get the most recent x[index] before updating it
    old_x_at_idx = (*x)[index];
    // Step 4: update x[index]
    (*x)[index] -= relaxation_step_size * S_i;
    // Step 5: update the maintained variable
    prox.update_cache_vars(old_x_at_idx, (*x)[index], index);
    return S_i;
  }
  
  // TODO: implement this
  void operator() (int index, double& S_i) {
  }

  void update(Vector& s, int range_start, int num_cords) {
    for (size_t i = 0; i < num_cords; ++i ) {
      (*x)[i+range_start] -= relaxation_step_size * s[i];
    }
  }
  
  void update (double s, int idx ) {
    (*x)[idx] -= relaxation_step_size * s;
  }

  void update_cache_vars(int rank, int num_threads) {
    prox.update_cache_vars(x, rank, num_threads);
  }
  
};


// gradient descent algorithm
template <typename Forward>
class GradientDescentAlgorithm : public SchemeInterface {
public:  
  Forward forward;
  Vector *x;
  double relaxation_step_size;

  GradientDescentAlgorithm(Vector* x_, Forward forward_) {
    x = x_;
    forward = forward_;
    relaxation_step_size = 1.;
  }

  void update_params(Params* params) {
    forward.update_step_size(params->get_step_size());
    relaxation_step_size = params->get_tmac_step_size();
  }

  // update x[index] and update the maintained variables
  // x^{k+1} = x^k + eta_k (\hat x^k - T \hat x^k)
  double operator() (int index) {

    // Step 1: read the old x[index]
    double old_x_at_idx = (*x)[index];

    // Step 2: local calculation
    double forward_grad_at_idx = forward(x, index);
    double S_i = old_x_at_idx - forward_grad_at_idx;

    // Step 3: get the most recent x[index]
    old_x_at_idx = (*x)[index];
    
    // Step 4: update x at index
    (*x)[index] -= relaxation_step_size * S_i;

    // Step 5: update the maintained variable Atx
    double diff = (*x)[index] - old_x_at_idx;
    forward.update_cache_vars(old_x_at_idx, (*x)[index], index);
    return S_i;
  }

  void operator() (int index, double& S_i) {
    // Step 1: read the old x[index]
    double old_x_at_idx = (*x)[index];
    // Step 2: local calculation
    double forward_grad_at_idx = forward(x, index);
    S_i = old_x_at_idx - forward_grad_at_idx;
  }

  void update(Vector& s, int range_start, int num_cords) {
    for (size_t i = 0; i < num_cords; ++i ) {
      (*x)[i+range_start] -= relaxation_step_size * s[i];
    }
  }
  
  void update (double s, int idx ) {
    (*x)[idx] -= relaxation_step_size * s;
  }
  
  //update rank worth of cache_vars based on num_threads
  void update_cache_vars(int rank, int num_threads) {
    forward.update_cache_vars(x, rank, num_threads);
  }
  
};


// forward backward splitting scheme
template <typename Forward, typename Backward>
class ForwardBackwardSplitting : public SchemeInterface {
public:  
  Forward forward;
  Backward backward;
  Vector* x;
  double relaxation_step_size;

  ForwardBackwardSplitting(Vector* x_, Forward forward_, Backward backward_) {
    x = x_;
    forward = forward_;
    backward = backward_;
    relaxation_step_size = 1.;
  }

  void update_params(Params* params) {
    // TODO: forward and backward might use different step sizes
    forward.update_step_size(params->get_step_size());
    backward.update_step_size(params->get_step_size());
    relaxation_step_size = params->get_tmac_step_size();
  }

  double operator() (int index) {
    // Step 1: read the old x[index]
    double old_x_at_idx = (*x)[index];    
    // Step 2: local calculation
    double forward_grad_at_idx = forward(x, index);
    double val = backward(forward_grad_at_idx, index);
    double S_i = old_x_at_idx - val;
    // Step 3: get the most recent x[index] before updating it
    old_x_at_idx = (*x)[index];
    // Step 4: update x at index 
    (*x)[index] -= relaxation_step_size * S_i;
    // Step 5: update the maintained variable Atx
    forward.update_cache_vars(old_x_at_idx, (*x)[index], index);
    return S_i;
  }

  void operator() (int index, double &S_i) {
    // Step 1: read the old x[index]
    double old_x_at_idx = (*x)[index];    
    // Step 2: local calculation
    double forward_grad_at_idx = forward(x, index);
    double val = backward(forward_grad_at_idx);
    S_i = old_x_at_idx - val;
  }

  void update(Vector& s, int range_start, int num_cords) {
    for (size_t i = 0; i < num_cords; ++i ) {
      (*x)[i+range_start] -= relaxation_step_size * s[i];
    }
  }
  
  void update (double s, int idx ) {
    (*x)[idx] -= relaxation_step_size * s;
  }

  //update rank worth of cache_vars based on num_threads
  void update_cache_vars(int rank, int num_threads) {
    forward.update_cache_vars(x, rank, num_threads);
  }

};


// backward forward splitting scheme
// WARNING: this won't work, if the forward step has maintained variables.
template <typename Backward, typename Forward>
class BackwardForwardSplitting : public SchemeInterface {
public:  
  Forward forward;
  Backward backward;
  Vector *x;
  Vector y; // each new operator will have a copy of this guy
  double relaxation_step_size;

  BackwardForwardSplitting(Vector* x_, Backward backward_, Forward forward_) {
    x = x_;
    forward = forward_;
    backward = backward_;
    relaxation_step_size = 1.;
    y.resize(x->size());
  }

  void update_params(Params* params) {
    // TODO: forward and backward might use different step sizes
    forward.update_step_size(params->get_step_size());
    backward.update_step_size(params->get_step_size());
    relaxation_step_size = params->get_tmac_step_size();
  }

  double operator() (int index) {
    // Step 1: get the old x[index]
    double old_x_at_idx = (*x)[index];
    // Step 2: local computation
    // first apply the backward operator
    backward(x, &y);
    // then apply the forward operator on y
    double forward_grad_at_idx = forward(&y, index);
    double S_i = old_x_at_idx - forward_grad_at_idx;
    // Step 3: get the most recent x[index]
    old_x_at_idx = (*x)[index];
    // Step 4: update x[index]
    (*x)[index] -= relaxation_step_size * S_i;
    // Step 5: update the maintained variables
    // TODO: update cached_variable of the backward operator
    return S_i;
  }

  // TODO: implement this for sync-operator
  void operator()(int index, double &S_i) {
  }

  void update(Vector& s, int range_start, int num_cords) {
    for (size_t i = 0; i < num_cords; ++i ) {
      (*x)[i+range_start] -= relaxation_step_size * s[i];
    }
  }
  
  void update (double s, int idx ) {
    (*x)[idx] -= relaxation_step_size * s;
  }
  
  // TODO: for sync-parallel
  void update_cache_vars ( int rank, int index ) {
    std::cerr << "functionality not provided." << std::endl;
    exit(EXIT_FAILURE);
  }

};


// Peaceman-Rachford Splitting
// x^{k+1} = (1 - eta_k) x^k + eta_k (2 * First - I )(2 * Second - I) (x^k)
// which can be simplified to the following
// y^k = Second(x^k)
// z^k = First(2 y^k - x^k)
// x^{k+1} = x^k + 2 eta_k (z^k - y^k)
template <typename First, typename Second>
class PeacemanRachfordSplitting : public SchemeInterface {
public:  
  First op1;
  Second op2; 
  Vector *x;
  Vector y; // each new operator will have a copy of this guy
  Vector z;
  double relaxation_step_size;

  PeacemanRachfordSplitting(Vector* x_, First op1_, Second op2_) {
    x = x_;
    op1 = op1_;
    op2 = op2_;
    relaxation_step_size = 1.;
    y.resize(x->size());
    z.resize(x->size());
  }

  void update_params(Params* params) {
    // TODO: forward and backward might use different step sizes
    op1.update_step_size(params->get_step_size());
    op2.update_step_size(params->get_step_size());
    relaxation_step_size = params->get_tmac_step_size();
  }

  double operator() (int index) {
    // Step 1: get the old x[index]
    double old_x_at_idx = (*x)[index];
    // Step 2: local computation
    op2(x, &y);
    // z = 2y - x
    z = y;
    scale(z, 2.);
    add(z, *x, -1.);
    // prox(z, i), it doesn't have to be type I
    double temp = op1(&z, index);
    // Step 3: get the most recent x[index]
    old_x_at_idx = (*x)[index];
    // Step 4: update x at index 
    (*x)[index] += 2 * relaxation_step_size * (temp - y[index]);
    // Step 5: update the maintained variables
    return temp-y[index];
  }

  // TODO: implement this for sync-operator
  void operator()(int index, double &S_i) {
  }

  void update(Vector& s, int range_start, int num_cords) {
    for (size_t i = 0; i < num_cords; ++i ) {
      (*x)[i+range_start] -= relaxation_step_size * s[i];
    }
  }
  
  void update (double s, int idx ) {
    (*x)[idx] -= relaxation_step_size * s;
  }

  void update_cache_vars (int rank, int index ) {
  }
  
};

// ADMM based on DRS Splitting
// which can be simplified to the following
// y^k_i = op2(x^k_i) = argmin_t(f(x^k_i, t))
// u^k_i = (prox_df(x))_i = x^k_i + gamma * y^k_i
// temp = 2 * u^k_i - x^k_i
// z^k_i =  op1(temp) = argmin_t(g(temp, t))
// v^k_i = (prox_dg(temp))_i = temp - gamma * z^k_i
// x^{k+1}_i = x^k + eta_k * (v^k_i - u^k_i)
// avrg += eta_k * (v^k_i - u^k_i)/N
template <typename First, typename Second, typename Third>
class DoglasRachfordSplittingAdmm : public SchemeInterface {
public:
    First op1;
    Second op2;
    Third op3;
    Vector *x;
    double relaxation_step_size;
    double weight = 1.;
    
    DoglasRachfordSplittingAdmm(Vector* x_, First op1_, Second op2_, Third op3_) {
        x = x_;
        op1 = op1_;
        op2 = op2_;
        op3 = op3_;
        relaxation_step_size = 1.;
    }
    
    void update_params(Params* params) {
        // TODO: forward and backward might use different step sizes
        op1.update_step_size(params->get_step_size());
        op2.update_step_size(params->get_step_size());
        op3.update_step_size(params->get_step_size());
        relaxation_step_size = params->get_tmac_step_size();
    }
    
    double operator() (int index) {
        // Step 0: get the old x[index]
        double old_x_at_idx = (*x)[index];
        // Step 1: y = op2(x) = argmin_t(f(x, t))
        double y = op2(old_x_at_idx);
        // Step 2: u = x + gamma * y
        double u = old_x_at_idx + weight * y;
        // Step 3: z = op1(temp) = argmin_t(g(temp, t)), temp = 2 * u - x
        double temp = 2. * u - (*x)[index];
        double z = op1(temp, index);
        // Step 4: v = (prox_dg(temp))_i = temp - gamma * z
        double v = temp - weight * z;
        // Step 5: update x at index
        // x += eta * (v - u)
        double ss = relaxation_step_size * (v - u);
        (*x)[index] += ss;
        // Step 6: update the maintained variables
        //avrg += eta * (v - u);
        op3.update_cache_vars(0, ss/x->size(), 0);
        return ss;
    }
    
    // TODO: implement this for sync-operator
    void operator()(int index, double &S_i) {
    }
    
    void update(Vector& s, int range_start, int num_cords) {
        for (size_t i = 0; i < num_cords; ++i ) {
            (*x)[i+range_start] -= relaxation_step_size * s[i];
        }
    }
    
    void update (double s, int idx ) {
        (*x)[idx] -= relaxation_step_size * s;
    }
    
    void update_cache_vars (int rank, int index ) {
    }
    
};



// ADMM based on DRS Splitting, the Vector form(each unit is a vector)
// which can be simplified to the following
// y^k_i = op2(x^k_i) = argmin_t(f(x^k_i, t))
// u^k_i = (prox_df(x))_i = x^k_i + gamma * y^k_i
// temp = 2 * u^k_i - x^k_i
// z^k_i =  op1(temp) = argmin_t(g(temp, t))
// v^k_i = (prox_dg(temp))_i = temp - gamma * z^k_i
// x^{k+1}_i = x^k + eta_k * (v^k_i - u^k_i)
// avrg += eta_k * (v^k_i - u^k_i)
template <typename First, typename Second, typename Third>
class DoglasRachfordSplittingAdmm_vector : public SchemeInterface {
public:
    int unit_len;
    First op1;
    Second op2;
    Third op3;
    vector<Vector> *x;
    Vector u_i;
    Vector v_i;
    double relaxation_step_size;
    double weight = 1.;
    
    DoglasRachfordSplittingAdmm_vector(vector<Vector>* x_, First op1_, Second op2_, Third op3_) {
        unit_len = (*x_)[0].size();
        x = x_;
        op1 = op1_;
        op2 = op2_;
        op3 = op3_;
        relaxation_step_size = 1.;
        u_i.resize(unit_len);
        v_i.resize(unit_len);
    }
    
    void update_params(Params* params) {
        // TODO: forward and backward might use different step sizes
        op1.update_step_size(params->get_step_size());
        op2.update_step_size(params->get_step_size());
        op3.update_step_size(params->get_step_size());
        relaxation_step_size = params->get_tmac_step_size();
        u_i.resize(unit_len);
        v_i.resize(unit_len);
    }
    
    double operator() (int index) {
        // Step 0: get the old x[index]
        Vector old_x_at_idx;
        old_x_at_idx.resize(unit_len);
        copy((*x)[index], old_x_at_idx, 0, (*x)[index].size());
        // Step 1: y = op2(x) = argmin_t(f(x, t))
        Vector y_i(unit_len, 0.);
        op2(&old_x_at_idx, &y_i);
        // Step 2: u = x + gamma * y
        copy(old_x_at_idx, u_i, 0, old_x_at_idx.size());
        add(u_i, y_i, weight);
        // Step 3: z = op1(temp) = argmin_t(g(temp, t)), temp = 2 * u - x
        copy(u_i, v_i, 0, u_i.size());
        scale(v_i, 2.);
        add(v_i, old_x_at_idx , -1.);
        Vector z_i(unit_len, 0.);
        copy(v_i, z_i, 0, v_i.size());
        op1(&z_i, index);
        // Step 4: v = (prox_dg(temp))_i = temp - gamma * z
        add(v_i, z_i, -weight);
        // Step 5: update x at index
        // x += eta * (v - u)
        add(v_i, u_i, -1.);
        scale(v_i, relaxation_step_size);
        //double ss = relaxation_step_size * (temp - y);
        add((*x)[index], v_i, 1.);
        // Step 6: update the maintained variables
        //avrg += eta * (v - u);
        op3.update_cache_vars(&v_i, 0, 0);
        return 0;
    }
    
    // TODO: implement this for sync-operator
    void operator()(int index, double &S_i) {
    }
    
    void update(Vector& s, int range_start, int num_cords) {
    }
    
    void update (double s, int idx ) {
    }
    
    void update_cache_vars (int rank, int index ) {
    }
    
};




#endif
