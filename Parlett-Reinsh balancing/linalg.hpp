#pragma once

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace f22 {

class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, double fill);

    std::size_t rows() const;
    std::size_t cols() const;

    double& operator()(std::size_t i, std::size_t j);
    const double& operator()(std::size_t i, std::size_t j) const;

    Matrix transpose() const;
    Matrix operator*(const Matrix& other) const;
    std::vector<double> matvec(const std::vector<double>& v) const;

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

using ResidualFn = std::function<std::vector<double>(const std::vector<double>&)>;

struct LMOptions {
    double lambda_init = 1e-3;
    double lambda_up = 10.0;
    double lambda_down = 0.1;
    int max_iterations = 200;
    double fd_eps = 1e-7;
    double grad_tol = 1e-12;
    double step_tol = 1e-14;
};

struct LMResult {
    std::vector<double> x;
    std::vector<double> residuals;
    double residual_norm = 0.0;
    bool converged = false;
};

std::vector<double> solve_linear(Matrix A, std::vector<double> b);

LMResult levenberg_marquardt(const ResidualFn& residual_fn, std::vector<double> x0,
                              const std::vector<double>& lower, const std::vector<double>& upper,
                              const LMOptions& opts = LMOptions{});

std::vector<std::complex<double>> real_eigenvalues(const Matrix& A, int max_iterations = 300);

}  // namespace f22
