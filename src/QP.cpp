#include "QP.hpp"
#include <iostream>

// QP problem constructor
QP::QP(Eigen::MatrixXd Qin, Eigen::MatrixXd qin, Eigen::MatrixXd Gin, Eigen::MatrixXd hin, Eigen::MatrixXd Ain, Eigen::MatrixXd bin){

    max_iterations = 20;

    n = Qin.cols();
    p = Gin.rows();
    m = Ain.rows();

    dim = n + p + p + m;

    Q = Qin;
    q = qin;
    G = Gin;
    h = hin;
    A = Ain;
    b = bin;

    Eigen::MatrixXd reg_dense(dim, dim);
    reg_dense.setZero();

    epsilon = 0.0000001;

    reg_dense.block(0,0, n+p, n+p).diagonal().array() += epsilon * 1.0;
    reg_dense.block(n+p, n+p, p+m, p+m).diagonal().array() -= epsilon * 1.0;

    reg_sparse = reg_dense.sparseView();
    rhs_aff.resize(dim);
    rhs_cor.resize(dim);
    rhs_cor.setZero();

    delta_aff.x.resize(n);
    delta_aff.s.resize(p);
    delta_aff.z.resize(p);
    delta_aff.y.resize(m);

    delta_cor.x.resize(n);
    delta_cor.s.resize(p);
    delta_cor.z.resize(p);
    delta_cor.y.resize(m);

    delta.x.resize(n);
    delta.s.resize(p);
    delta.z.resize(p);
    delta.y.resize(m);
}

// form initial KKT matrix from problem data
void QP::form_KKT(){
    int my_rows  = A.rows();
    int my_cols = A.cols();
    KKT.resize(dim, dim);
    KKT.setZero();

    KKT.block(0,0,n,n) = Q;
    KKT.block(0, n + p, n, p) = G.transpose();
    KKT.block(0, n + p + p, n, m) = A.transpose();
    KKT.block(n, n + p, p, p) = Eigen::MatrixXd::Identity(p,p);

    KKT.block(n + p, 0, p, n) = G;
    KKT.block(n + p, n, p, p) = Eigen::MatrixXd::Identity(p,p);
    KKT.block(n + p + p, 0, m , n) = A;
}

// update complementarity variables in KKT matrix
void QP::update_KKT(){
    KKT.block(n, n, p, p) = z.cwiseQuotient(s).asDiagonal();
}

// regularized the diagonal of KKT matrix
void QP::regularize_KKT(){
    KKT_sparse_reg = KKT.sparseView();
    KKT_sparse_reg += reg_sparse;
}

// Solve KKT system via iterative refinement
void QP::solve_KKT(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> &solver, Eigen::VectorXd &rhs, Delta &delta){
    Eigen::VectorXd l(dim);
    Eigen::VectorXd dl(dim);
    solver.factorize(KKT_sparse_reg);
    l = solver.solve(rhs);
    
    while ((rhs - KKT*l).norm() > 0.000001){
        dl = solver.solve(rhs - KKT*l);
        l += dl;
    }

    delta.x = l.segment(0, n);
    delta.s = l.segment(n, p);
    delta.z = l.segment(n+p, p);
    delta.y = l.segment(n+p+p, m);
}

// update right-hand side for affine scaling direction
void QP::update_rhs_aff(){
    rhs_aff.segment(0, n) = -(A.transpose() * y + G.transpose()*z + Q*x + q);
    rhs_aff.segment(n, p) = -z;
    rhs_aff.segment(n + p, p) = -(G*x + s - h);
    rhs_aff.segment(n + p + p, m) = -(A*x - b);
}

// update right-hand side for corrector step
void QP::update_rhs_cor(){
    Eigen::VectorXd s_inv = s.array().inverse();
    Eigen::MatrixXd S_inv = s_inv.asDiagonal();
    rhs_cor.segment(n, p) = S_inv*(sigma*mu*Eigen::VectorXd::Ones(p) - delta_aff.s.asDiagonal()*delta_aff.z);
}

// update surrogate duality gap
void QP::update_mu(){
    mu = s.dot(z) / p;
}

// gets maximum step length to remain in the nonnegative orthant
double QP::get_max_step_length(Eigen::VectorXd x, Eigen::VectorXd dx){
    double min = 100;
    double temp;

    for(int i = 0; i < x.size(); i++){   

        if (dx(i) < 0){
            temp = -x(i) / dx(i);
            if (temp < min){
                min = temp;
            }
        }
    }

    return fmin(1.0, min);
}

// update centering parameter
void QP::update_sigma(){
    double alpha_pri = get_max_step_length(s, delta.s);
    double alpha_dua = get_max_step_length(z, delta.z);
    double alpha = fmin(alpha_pri, alpha_dua);

    sigma = pow(((s + alpha*delta_aff.s).dot(z + alpha*delta_aff.z)) / (s.dot(z)), 3);
}

// initialize variables to the interior of the feasible set
void QP::initialize_vars(){
    Eigen::MatrixXd KKT_init;
    Eigen::VectorXd rhs_init;

    KKT_init.resize(n + p + m , n + p + m);
    KKT_init.setZero();
    KKT_init.block(0, 0, n, n) = Q;
    KKT_init.block(0, n, n, p) = G.transpose();
    KKT_init.block(0, n + p, n, m) = A.transpose();

    KKT_init.block(n, 0, p, n) = G;
    KKT_init.block(n + p, 0, m, n) = A;
    KKT_init.block(n, n, p, p) = -1*Eigen::MatrixXd::Identity(p,p);

    rhs_init.resize(n + p + m);
    rhs_init.segment(0, n) = -q;
    rhs_init.segment(n, p) = h;
    rhs_init.segment(n+p, m) = b;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_init;
    solver_init.compute(KKT_init.sparseView());

    Eigen::VectorXd sol_init = solver_init.solve(rhs_init);

    x = sol_init.segment(0,n);
    y = sol_init.segment(n+p, m);

    Eigen::VectorXd z_init = sol_init.segment(n, p);

    double ap = z_init.maxCoeff();
    double ad = (-z_init).maxCoeff(); 

    if(ap < 0){
        s = -z_init;
    }
    else{
        s = -z_init + (1 + ap)*Eigen::VectorXd::Ones(p);
    }

    if(ad < 0){
        z = z_init;
    }
    else{
        z = z_init + (1 + ad)*Eigen::VectorXd::Ones(p);
    }
}

// final solver algorithm
void QP::solve(){

    // initialize variables and form the initial KKT matrix
    initialize_vars();
    form_KKT();
    update_KKT();
    regularize_KKT();

    // initilize LDLT solver and analyze sparsity pattern
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(KKT_sparse_reg);
    
    for (int i = 0; i< max_iterations; i++){
        update_KKT();
        regularize_KKT();

        // predictor step
        update_rhs_aff();
        solve_KKT(solver, rhs_aff, delta_aff);
        
        // corrector step
        update_mu();
        update_sigma();
        update_rhs_cor();
        solve_KKT(solver, rhs_cor, delta_cor);

        // combine prediction and corrector steps
        delta.x = delta_aff.x + delta_cor.x;
        delta.s = delta_aff.s + delta_cor.s;
        delta.z = delta_aff.z + delta_cor.z;
        delta.y = delta_aff.y + delta_cor.y;

        double alpha;
        alpha = fmin(1.0, 0.99*fmin(get_max_step_length(s, delta.s), get_max_step_length(z, delta.z)));

        x += alpha * delta.x;
        s += alpha * delta.s;
        z += alpha * delta.z;
        y += alpha * delta.y;
    }
}

double QP::get_opt_value(){
    return 0.5*x.transpose()*Q*x + q.dot(x);
}