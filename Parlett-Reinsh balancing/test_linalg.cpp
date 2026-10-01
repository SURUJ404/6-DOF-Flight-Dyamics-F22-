// test_linalg.cpp
#include "linalg.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <complex>
#include <vector>

using namespace f22;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                     \
    do {                                               \
        ++tests_run;                                   \
        printf("  %-50s ", #name);                     \
        try {                                          \
            test_##name();                             \
            ++tests_passed;                            \
            printf("PASS\n");                          \
        } catch (const std::exception& e) {            \
            printf("FAIL: %s\n", e.what());            \
        } catch (...) {                                \
            printf("FAIL: unknown exception\n");       \
        }                                              \
    } while (0)

#define ASSERT_EQ(a, b)                                                    \
    do {                                                                   \
        if ((a) != (b)) {                                                  \
            throw std::runtime_error(                                       \
                std::string("ASSERT_EQ failed: ") + #a + " != " + #b);     \
        }                                                                  \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                             \
    do {                                                                   \
        if (std::fabs((a) - (b)) > (eps)) {                                \
            throw std::runtime_error(                                       \
                std::string("ASSERT_NEAR failed: |") + #a " - " #b "| > " #eps); \
        }                                                                  \
    } while (0)

#define ASSERT_TRUE(expr)                                                  \
    do {                                                                   \
        if (!(expr)) {                                                     \
            throw std::runtime_error(                                       \
                std::string("ASSERT_TRUE failed: ") + #expr);              \
        }                                                                  \
    } while (0)

// =========================================================================
// Matrix basic tests
// =========================================================================
void test_matrix_default_ctor() {
    Matrix m;
    ASSERT_EQ(m.rows(), 0u);
    ASSERT_EQ(m.cols(), 0u);
}

void test_matrix_fill_ctor() {
    Matrix m(3, 4, 2.5);
    ASSERT_EQ(m.rows(), 3u);
    ASSERT_EQ(m.cols(), 4u);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(m(i, j), 2.5, 1e-15);
}

void test_matrix_transpose() {
    Matrix m(2, 3);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    Matrix T = m.transpose();
    ASSERT_EQ(T.rows(), 3u);
    ASSERT_EQ(T.cols(), 2u);
    ASSERT_NEAR(T(0, 0), 1, 1e-15);
    ASSERT_NEAR(T(0, 1), 4, 1e-15);
    ASSERT_NEAR(T(1, 0), 2, 1e-15);
    ASSERT_NEAR(T(1, 1), 5, 1e-15);
    ASSERT_NEAR(T(2, 0), 3, 1e-15);
    ASSERT_NEAR(T(2, 1), 6, 1e-15);
}

void test_matrix_multiply() {
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    Matrix B(2, 2);
    B(0, 0) = 5; B(0, 1) = 6;
    B(1, 0) = 7; B(1, 1) = 8;
    Matrix C = A * B;
    ASSERT_NEAR(C(0, 0), 19, 1e-12);
    ASSERT_NEAR(C(0, 1), 22, 1e-12);
    ASSERT_NEAR(C(1, 0), 43, 1e-12);
    ASSERT_NEAR(C(1, 1), 50, 1e-12);
}

void test_matrix_matvec() {
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    std::vector<double> v = {5, 6};
    std::vector<double> y = A.matvec(v);
    ASSERT_EQ(y.size(), 2u);
    ASSERT_NEAR(y[0], 17, 1e-12);
    ASSERT_NEAR(y[1], 39, 1e-12);
}

// =========================================================================
// solve_linear tests
// =========================================================================
void test_solve_identity() {
    Matrix I(3, 3);
    I(0, 0) = 1; I(1, 1) = 1; I(2, 2) = 1;
    std::vector<double> b = {1, 2, 3};
    auto x = solve_linear(I, b);
    ASSERT_NEAR(x[0], 1, 1e-12);
    ASSERT_NEAR(x[1], 2, 1e-12);
    ASSERT_NEAR(x[2], 3, 1e-12);
}

void test_solve_2x2() {
    Matrix A(2, 2);
    A(0, 0) = 2; A(0, 1) = 1;
    A(1, 0) = 5; A(1, 1) = 7;
    std::vector<double> b = {11, 13};
    auto x = solve_linear(A, b);
    // solution: x = [59/9, -17/9]
    ASSERT_NEAR(x[0], 59.0 / 9.0, 1e-10);
    ASSERT_NEAR(x[1], -17.0 / 9.0, 1e-10);
}

void test_solve_3x3() {
    Matrix A(3, 3);
    A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = -1;
    A(1, 0) = -3; A(1, 1) = -1; A(1, 2) = 2;
    A(2, 0) = -2; A(2, 1) = 1; A(2, 2) = 2;
    std::vector<double> b = {8, -11, -3};
    auto x = solve_linear(A, b);
    ASSERT_NEAR(x[0], 2, 1e-10);
    ASSERT_NEAR(x[1], 3, 1e-10);
    ASSERT_NEAR(x[2], -1, 1e-10);
}

void test_solve_identity_1x1() {
    Matrix A(1, 1);
    A(0, 0) = 5;
    std::vector<double> b = {10};
    auto x = solve_linear(A, b);
    ASSERT_NEAR(x[0], 2.0, 1e-12);
}

void test_solve_singular() {
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 4;
    std::vector<double> b = {3, 6};
    bool threw = false;
    try {
        solve_linear(A, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

void test_solve_dimension_mismatch() {
    Matrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 1;
    std::vector<double> b = {1, 2, 3};
    bool threw = false;
    try {
        solve_linear(A, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

void test_solve_identity_4x4() {
    Matrix I(4, 4);
    for (std::size_t i = 0; i < 4; ++i) I(i, i) = 1.0;
    std::vector<double> b = {1, -2, 3, -4};
    auto x = solve_linear(I, b);
    for (std::size_t i = 0; i < 4; ++i) ASSERT_NEAR(x[i], b[i], 1e-12);
}

void test_solve_random_5x5() {
    // Solve a known system: A * x = b where x = {1,2,3,4,5}
    Matrix A(5, 5);
    // Use a well-conditioned matrix (Vandermonde-like)
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            A(i, j) = std::pow(static_cast<double>(i + 1), static_cast<double>(j));
    std::vector<double> x_true = {1, 2, 3, 4, 5};
    auto b = A.matvec(x_true);
    auto x = solve_linear(A, b);
    for (std::size_t i = 0; i < 5; ++i) ASSERT_NEAR(x[i], x_true[i], 1e-8);
}

// =========================================================================
// Levenberg-Marquardt tests
// =========================================================================
void test_lm_trivial() {
    // Residual is already zero at x* = {1, 2}
    ResidualFn fn = [](const std::vector<double>& x) -> std::vector<double> {
        return {x[0] - 1.0, x[1] - 2.0};
    };
    std::vector<double> lower = {-100, -100};
    std::vector<double> upper = {100, 100};
    auto result = levenberg_marquardt(fn, {0.5, 1.5}, lower, upper);
    ASSERT_NEAR(result.x[0], 1.0, 1e-6);
    ASSERT_NEAR(result.x[1], 2.0, 1e-6);
    ASSERT_NEAR(result.residual_norm, 0.0, 1e-6);
}

void test_lm_linear_fit() {
    // Fit y = a*x + b to data points (1,3), (2,5), (3,7) -> a=2, b=1
    // residual_i = (a*x_i + b) - y_i
    std::vector<double> xdata = {1, 2, 3};
    std::vector<double> ydata = {3, 5, 7};
    ResidualFn fn = [&](const std::vector<double>& p) -> std::vector<double> {
        std::vector<double> r(3);
        for (int i = 0; i < 3; ++i)
            r[i] = (p[0] * xdata[i] + p[1]) - ydata[i];
        return r;
    };
    std::vector<double> lower = {-100, -100};
    std::vector<double> upper = {100, 100};
    auto result = levenberg_marquardt(fn, {0, 0}, lower, upper);
    ASSERT_NEAR(result.x[0], 2.0, 1e-4);
    ASSERT_NEAR(result.x[1], 1.0, 1e-4);
}

void test_lm_quadratic_fit() {
    // Fit y = a*x^2 to points (1,1), (2,4), (3,9) -> a=1
    std::vector<double> xdata = {1, 2, 3};
    std::vector<double> ydata = {1, 4, 9};
    ResidualFn fn = [&](const std::vector<double>& p) -> std::vector<double> {
        std::vector<double> r(3);
        for (int i = 0; i < 3; ++i)
            r[i] = (p[0] * xdata[i] * xdata[i]) - ydata[i];
        return r;
    };
    std::vector<double> lower = {-100};
    std::vector<double> upper = {100};
    auto result = levenberg_marquardt(fn, {0.5}, lower, upper);
    ASSERT_NEAR(result.x[0], 1.0, 1e-4);
}

void test_lm_bounds_active() {
    // Minimize (x-10)^2 but x clamped to [0, 3] -> optimal at x=3
    ResidualFn fn = [](const std::vector<double>& x) -> std::vector<double> {
        return {x[0] - 10.0};
    };
    std::vector<double> lower = {0};
    std::vector<double> upper = {3};
    auto result = levenberg_marquardt(fn, {0}, lower, upper);
    ASSERT_NEAR(result.x[0], 3.0, 1e-4);
}

void test_lm_rosenbrock() {
    // Rosenbrock-like: minimize (1 - x)^2 + 100*(y - x^2)^2
    // residual = [1-x, 10*(y - x^2)] -> zero at (1,1)
    ResidualFn fn = [](const std::vector<double>& p) -> std::vector<double> {
        return {1.0 - p[0], 10.0 * (p[1] - p[0] * p[0])};
    };
    std::vector<double> lower = {-10, -10};
    std::vector<double> upper = {10, 10};
    auto result = levenberg_marquardt(fn, {-1, 1}, lower, upper);
    ASSERT_NEAR(result.x[0], 1.0, 1e-4);
    ASSERT_NEAR(result.x[1], 1.0, 1e-4);
    ASSERT_NEAR(result.residual_norm, 0.0, 1e-4);
}

void test_lm_convergence_flag() {
    ResidualFn fn = [](const std::vector<double>& x) -> std::vector<double> {
        return {x[0] - 1.0};
    };
    std::vector<double> lower = {-100};
    std::vector<double> upper = {100};
    auto result = levenberg_marquardt(fn, {0.5}, lower, upper);
    ASSERT_TRUE(result.converged);
}

// =========================================================================
// Eigenvalue tests
// =========================================================================
void test_eigen_1x1() {
    Matrix A(1, 1);
    A(0, 0) = 7.0;
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 1u);
    ASSERT_NEAR(eigs[0].real(), 7.0, 1e-12);
    ASSERT_NEAR(eigs[0].imag(), 0.0, 1e-12);
}

void test_eigen_2x2_real() {
    // Diagonal matrix -> eigenvalues on diagonal
    Matrix A(2, 2);
    A(0, 0) = 3; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 5;
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 2u);
    ASSERT_NEAR(eigs[0].real(), 3.0, 1e-10);
    ASSERT_NEAR(eigs[1].real(), 5.0, 1e-10);
}

void test_eigen_2x2_complex() {
    // Rotation matrix with angle pi/4 -> eigenvalues e^{+/- i pi/4}
    double c = std::cos(M_PI / 4), s = std::sin(M_PI / 4);
    Matrix A(2, 2);
    A(0, 0) = c;  A(0, 1) = -s;
    A(1, 0) = s;  A(1, 1) = c;
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 2u);
    // Both should have |lambda| = 1
    double mag0 = std::abs(eigs[0]);
    double mag1 = std::abs(eigs[1]);
    ASSERT_NEAR(mag0, 1.0, 1e-8);
    ASSERT_NEAR(mag1, 1.0, 1e-8);
    // Imaginary parts should be +/- sin(pi/4)
    ASSERT_NEAR(std::fabs(eigs[0].imag()), s, 1e-8);
}

void test_eigen_3x3_diagonal() {
    Matrix A(3, 3);
    A(0, 0) = 1; A(1, 1) = 2; A(2, 2) = 3;
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 3u);
    ASSERT_NEAR(eigs[0].real(), 1.0, 1e-10);
    ASSERT_NEAR(eigs[1].real(), 2.0, 1e-10);
    ASSERT_NEAR(eigs[2].real(), 3.0, 1e-10);
}

void test_eigen_3x3_symmetric() {
    // Symmetric matrix -> all real eigenvalues
    Matrix A(3, 3);
    A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = 0;
    A(1, 0) = -1; A(1, 1) = 2; A(1, 2) = -1;
    A(2, 0) = 0; A(2, 1) = -1; A(2, 2) = 2;
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 3u);
    // Eigenvalues of tridiagonal [2,-1,0; -1,2,-1; 0,-1,2] are 2-sqrt(2), 2, 2+sqrt(2)
    ASSERT_NEAR(eigs[0].real(), 2.0 - std::sqrt(2.0), 1e-8);
    ASSERT_NEAR(eigs[1].real(), 2.0, 1e-8);
    ASSERT_NEAR(eigs[2].real(), 2.0 + std::sqrt(2.0), 1e-8);
}

void test_eigen_trace_determinant_3x3() {
    // For a 3x3 matrix, sum of eigenvalues = trace, product = det
    Matrix A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 0; A(1, 1) = 4; A(1, 2) = 5;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 6;
    // trace = 11, det = 24, eigenvalues are 1, 4, 6 (upper triangular)
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 3u);
    double trace = 0;
    for (auto& e : eigs) trace += e.real();
    ASSERT_NEAR(trace, 11.0, 1e-8);
}

void test_eigen_4x4_identity() {
    Matrix I(4, 4);
    for (std::size_t i = 0; i < 4; ++i) I(i, i) = 1.0;
    auto eigs = real_eigenvalues(I);
    ASSERT_EQ(eigs.size(), 4u);
    for (auto& e : eigs) {
        ASSERT_NEAR(e.real(), 1.0, 1e-10);
        ASSERT_NEAR(e.imag(), 0.0, 1e-10);
    }
}

void test_eigen_5x5_symmetric() {
    // 5x5 symmetric tridiagonal with known eigenvalues:
    // A = diag(2) - diag(1, offset=1) - diag(1, offset=-1)
    // Eigenvalues: 2 - 2*cos(k*pi/6), k=1..5
    Matrix A(5, 5);
    for (std::size_t i = 0; i < 5; ++i) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < 4) A(i, i + 1) = -1.0;
    }
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 5u);
    for (int k = 1; k <= 5; ++k) {
        double expected = 2.0 - 2.0 * std::cos(k * M_PI / 6.0);
        bool found = false;
        for (auto& e : eigs) {
            if (std::fabs(e.real() - expected) < 1e-6) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found);
    }
}

void test_eigen_empty() {
    Matrix A(0, 0);
    auto eigs = real_eigenvalues(A);
    ASSERT_EQ(eigs.size(), 0u);
}

// =========================================================================
// Main
// =========================================================================
int main() {
    printf("=== Matrix Tests ===\n");
    TEST(matrix_default_ctor);
    TEST(matrix_fill_ctor);
    TEST(matrix_transpose);
    TEST(matrix_multiply);
    TEST(matrix_matvec);

    printf("\n=== solve_linear Tests ===\n");
    TEST(solve_identity);
    TEST(solve_1x1);
    TEST(solve_2x2);
    TEST(solve_3x3);
    TEST(solve_4x4);
    TEST(solve_random_5x5);
    TEST(solve_singular);
    TEST(solve_dimension_mismatch);

    printf("\n=== levenberg_marquardt Tests ===\n");
    TEST(lm_trivial);
    TEST(lm_linear_fit);
    TEST(lm_quadratic_fit);
    TEST(lm_bounds_active);
    TEST(lm_rosenbrock);
    TEST(lm_convergence_flag);

    printf("\n=== real_eigenvalues Tests ===\n");
    TEST(eigen_empty);
    TEST(eigen_1x1);
    TEST(eigen_2x2_real);
    TEST(eigen_2x2_complex);
    TEST(eigen_3x3_diagonal);
    TEST(eigen_3x3_symmetric);
    TEST(eigen_trace_determinant_3x3);
    TEST(eigen_4x4_identity);
    TEST(eigen_5x5_symmetric);

    printf("\n========================================\n");
    printf("Results: %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
