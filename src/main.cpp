#include "QP.hpp"
#include <iostream>

int main(){

    // problem data
    Eigen::MatrixXd Q(3, 3);
    Q << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    Q = Q.transpose() * Q;

    Eigen::VectorXd q(3);
    q << 1, 2, 3;

    Eigen::MatrixXd G(3, 3);
    G << 2, 5, 3, 7, 1, 4, 2, 6, 3;

    Eigen::VectorXd h(3);
    h << 4, 5, 8;

    Eigen::MatrixXd A(1, 3);
    A << 4, 9, 5;

    Eigen::VectorXd b(1);
    b << 1;

    // solver
    QP prob(Q, q, G, h, A, b);
    prob.solve();

    std::cout << "optimal x:\n" << prob.x << "\n";

    std::cout << "optimal objective value: " << prob.get_opt_value() << "\n";
}