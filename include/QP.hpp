#pragma once
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "structs.hpp"

class QP {
private:
    //problem data
    Eigen::MatrixXd Q;
    Eigen::VectorXd q;
    Eigen::MatrixXd G;
    Eigen::VectorXd h;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    //sparse representation
    Eigen::SparseMatrix<double> Q_sp;
    Eigen::SparseMatrix<double> G_sp;
    Eigen::SparseMatrix<double> A_sp;

    //problem size
    int m;
    int n;
    int p;
    int dim;

    //decision variables
    Eigen::VectorXd s;
    Eigen::VectorXd z;
    Eigen::VectorXd y;

    //KKT variables
    Eigen::MatrixXd KKT;
    Eigen::SparseMatrix<double> KKT_sparse_reg;
    Eigen::VectorXd rhs_aff;
    Eigen::VectorXd rhs_cor;

    Delta delta_aff;
    Delta delta_cor;
    Delta delta;

    Eigen::SparseMatrix<double> reg_sparse;

    double sigma;
    double mu;
    double epsilon;
    int max_iterations;

public:
    Eigen::VectorXd x;
    QP(Eigen::MatrixXd Qin, Eigen::MatrixXd qin, Eigen::MatrixXd Gin, Eigen::MatrixXd hin, Eigen::MatrixXd Ain, Eigen::MatrixXd bin);
    void form_KKT();
    void update_KKT();
    void regularize_KKT();
    void update_rhs_aff();
    void update_rhs_cor();
    void solve_KKT(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &solver, Eigen::VectorXd &rhs, Delta &x);
    void initialize_vars();

    void update_mu();
    double get_max_step_length(Eigen::VectorXd x, Eigen::VectorXd dx);
    void update_sigma();
    void solve();
    double get_opt_value();
};