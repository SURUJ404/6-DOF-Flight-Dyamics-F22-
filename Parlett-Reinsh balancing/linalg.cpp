// linalg.cpp
#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace f22 {

// ===========================================================================
// Matrix
// ===========================================================================
Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

std::size_t Matrix::rows() const { return rows_; }
std::size_t Matrix::cols() const { return cols_; }

double& Matrix::operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
const double& Matrix::operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

Matrix Matrix::transpose() const {
    Matrix T(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < cols_; ++j) T(j, i) = (*this)(i, j);
    return T;
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix C(rows_, other.cols_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < other.cols_; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < cols_; ++k) s += (*this)(i, k) * other(k, j);
            C(i, j) = s;
        }
    return C;
}

std::vector<double> Matrix::matvec(const std::vector<double>& v) const {
    std::vector<double> y(rows_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < cols_; ++j) s += (*this)(i, j) * v[j];
        y[i] = s;
    }
    return y;
}

// ===========================================================================
// Gaussian elimination with partial pivoting
// ===========================================================================
std::vector<double> solve_linear(Matrix A, std::vector<double> b) {
    const std::size_t n = A.rows();
    if (A.cols() != n || b.size() != n) {
        throw std::invalid_argument("solve_linear: dimension mismatch");
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        double best = std::fabs(A(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            const double v = std::fabs(A(i, k));
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best < 1e-300) {
            throw std::runtime_error("solve_linear: singular matrix");
        }
        if (piv != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(A(k, j), A(piv, j));
            std::swap(b[k], b[piv]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = A(i, k) / A(k, k);
            if (f == 0.0) continue;
            for (std::size_t j = k; j < n; ++j) A(i, j) -= f * A(k, j);
            b[i] -= f * b[k];
        }
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= A(i, j) * x[j];
        x[i] = s / A(i, i);
    }
    return x;
}

// ===========================================================================
// Bounded Levenberg-Marquardt
// ===========================================================================
namespace {
double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double vi : v) s += vi * vi;
    return std::sqrt(s);
}

std::vector<double> clamp_to_bounds(std::vector<double> x, const std::vector<double>& lower,
                                     const std::vector<double>& upper) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::clamp(x[i], lower[i], upper[i]);
    return x;
}
}  // namespace

LMResult levenberg_marquardt(const ResidualFn& residual_fn, std::vector<double> x0,
                              const std::vector<double>& lower, const std::vector<double>& upper,
                              const LMOptions& opts) {
    const std::size_t n = x0.size();
    std::vector<double> x = clamp_to_bounds(x0, lower, upper);
    std::vector<double> r = residual_fn(x);
    const std::size_t m = r.size();
    double cost = norm2(r);
    double lambda = opts.lambda_init;

    LMResult result;
    for (int iter = 0; iter < opts.max_iterations; ++iter) {
        Matrix J(m, n);
        for (std::size_t j = 0; j < n; ++j) {
            const double step = opts.fd_eps * std::max(1.0, std::fabs(x[j]));
            std::vector<double> xp = x, xm = x;
            xp[j] += step;
            xm[j] -= step;
            const std::vector<double> rp = residual_fn(xp);
            const std::vector<double> rm = residual_fn(xm);
            for (std::size_t i = 0; i < m; ++i) J(i, j) = (rp[i] - rm[i]) / (2.0 * step);
        }

        const Matrix Jt = J.transpose();
        Matrix JtJ = Jt * J;
        std::vector<double> Jtr = Jt.matvec(r);

        const double grad_norm = norm2(Jtr);
        if (grad_norm < opts.grad_tol) {
            result.converged = true;
            break;
        }

        bool accepted = false;
        for (int inner = 0; inner < 30; ++inner) {
            Matrix damped = JtJ;
            for (std::size_t i = 0; i < n; ++i) damped(i, i) += lambda * std::max(JtJ(i, i), 1e-12);
            std::vector<double> rhs(n);
            for (std::size_t i = 0; i < n; ++i) rhs[i] = -Jtr[i];

            std::vector<double> delta;
            try {
                delta = solve_linear(damped, rhs);
            } catch (const std::runtime_error&) {
                lambda *= opts.lambda_up;
                continue;
            }

            std::vector<double> x_trial(n);
            for (std::size_t i = 0; i < n; ++i) x_trial[i] = x[i] + delta[i];
            x_trial = clamp_to_bounds(x_trial, lower, upper);

            const std::vector<double> r_trial = residual_fn(x_trial);
            const double cost_trial = norm2(r_trial);

            if (cost_trial < cost) {
                const double step_norm = norm2(delta);
                x = x_trial;
                r = r_trial;
                cost = cost_trial;
                lambda *= opts.lambda_down;
                accepted = true;
                if (step_norm < opts.step_tol) {
                    result.converged = true;
                }
                break;
            }
            lambda *= opts.lambda_up;
        }

        if (!accepted || result.converged) break;
    }

    result.x = x;
    result.residuals = r;
    result.residual_norm = cost;
    return result;
}

// ===========================================================================
// Real matrix eigenvalues: Householder Hessenberg reduction + implicit
// double-shift QR ("Francis") algorithm.
// ===========================================================================
namespace {

struct Reflector {
    double w[3] = {0, 0, 0};
    double beta = 0.0;
    int len = 0;
};

Reflector make_householder(const double* x, int len) {
    Reflector h;
    h.len = len;
    double norm_sq = 0.0;
    for (int i = 0; i < len; ++i) norm_sq += x[i] * x[i];
    const double norm = std::sqrt(norm_sq);
    if (norm < 1e-300) {
        h.beta = 0.0;
        return h;
    }
    double w[3];
    for (int i = 0; i < len; ++i) w[i] = x[i];
    const double sign = (x[0] >= 0.0) ? 1.0 : -1.0;
    w[0] = x[0] + sign * norm;
    double wnorm_sq = 0.0;
    for (int i = 0; i < len; ++i) wnorm_sq += w[i] * w[i];
    if (wnorm_sq < 1e-300) {
        h.beta = 0.0;
        return h;
    }
    for (int i = 0; i < len; ++i) h.w[i] = w[i];
    h.beta = 2.0 / wnorm_sq;
    return h;
}

void apply_left(Matrix& H, const Reflector& h, std::size_t row0, std::size_t col0,
                 std::size_t col1) {
    if (h.beta == 0.0) return;
    for (std::size_t j = col0; j <= col1; ++j) {
        double dot = 0.0;
        for (int i = 0; i < h.len; ++i) dot += h.w[i] * H(row0 + i, j);
        const double f = h.beta * dot;
        for (int i = 0; i < h.len; ++i) H(row0 + i, j) -= f * h.w[i];
    }
}

void apply_right(Matrix& H, const Reflector& h, std::size_t row0, std::size_t row1,
                  std::size_t col0) {
    if (h.beta == 0.0) return;
    for (std::size_t i = row0; i <= row1; ++i) {
        double dot = 0.0;
        for (int k = 0; k < h.len; ++k) dot += h.w[k] * H(i, col0 + k);
        const double f = h.beta * dot;
        for (int k = 0; k < h.len; ++k) H(i, col0 + k) -= f * h.w[k];
    }
}

void balance(Matrix& A) {
    const std::size_t n = A.rows();
    if (n < 2) return;
    constexpr double kRadix = 2.0;
    constexpr double kRadixSq = kRadix * kRadix;

    bool converged = false;
    while (!converged) {
        converged = true;
        for (std::size_t i = 0; i < n; ++i) {
            double c = 0.0, r = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                c += std::fabs(A(j, i));
                r += std::fabs(A(i, j));
            }
            if (c == 0.0 || r == 0.0) continue;

            double f = 1.0;
            double s = c + r;
            while (c < r / kRadix) {
                c *= kRadixSq;
                f *= kRadix;
                r /= kRadix;
            }
            while (c >= r * kRadix) {
                c /= kRadixSq;
                f /= kRadix;
                r *= kRadix;
            }

            if ((c + r) < 0.95 * s) {
                converged = false;
                const double f_inv = 1.0 / f;
                for (std::size_t j = 0; j < n; ++j) {
                    A(i, j) *= f_inv;
                    A(j, i) *= f;
                }
            }
        }
    }
}

void hessenberg_reduce(Matrix& H) {
    const std::size_t n = H.rows();
    if (n < 3) return;
    for (std::size_t k = 0; k + 2 < n; ++k) {
        const int len = static_cast<int>(n - k - 1);
        std::vector<double> col(len);
        for (int i = 0; i < len; ++i) col[i] = H(k + 1 + i, k);

        double norm_sq = 0.0;
        for (double v : col) norm_sq += v * v;
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-300) continue;

        const double sign = (col[0] >= 0.0) ? 1.0 : -1.0;
        std::vector<double> w = col;
        w[0] += sign * norm;
        double wnorm_sq = 0.0;
        for (double v : w) wnorm_sq += v * v;
        if (wnorm_sq < 1e-300) continue;
        const double beta = 2.0 / wnorm_sq;

        for (std::size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = 0; i < len; ++i) dot += w[i] * H(k + 1 + i, j);
            const double f = beta * dot;
            for (int i = 0; i < len; ++i) H(k + 1 + i, j) -= f * w[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            double dot = 0.0;
            for (int j = 0; j < len; ++j) dot += w[j] * H(i, k + 1 + j);
            const double f = beta * dot;
            for (int j = 0; j < len; ++j) H(i, k + 1 + j) -= f * w[j];
        }
    }
}

void eig2x2(double a, double b, double c, double d, std::complex<double>& l1,
            std::complex<double>& l2) {
    const double tr = a + d;
    const double det = a * d - b * c;
    const double disc = tr * tr - 4.0 * det;
    if (disc >= 0.0) {
        const double s = std::sqrt(disc);
        l1 = std::complex<double>((tr + s) / 2.0, 0.0);
        l2 = std::complex<double>((tr - s) / 2.0, 0.0);
    } else {
        const double s = std::sqrt(-disc);
        l1 = std::complex<double>(tr / 2.0, s / 2.0);
        l2 = std::complex<double>(tr / 2.0, -s / 2.0);
    }
}

}  // namespace

std::vector<std::complex<double>> real_eigenvalues(const Matrix& A, int max_iterations) {
    const std::size_t n = A.rows();
    std::vector<std::complex<double>> eigenvalues;
    if (n == 0) return eigenvalues;
    if (n == 1) {
        eigenvalues.emplace_back(A(0, 0), 0.0);
        return eigenvalues;
    }

    Matrix H = A;
    balance(H);
    hessenberg_reduce(H);

    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) scale += std::fabs(H(i, j));
    if (scale < 1e-300) scale = 1.0;
    const double eps = 1e-13;

    long hi = static_cast<long>(n) - 1;
    long low = 0;
    int iterations_since_deflation = 0;

    while (hi >= 0) {
        long l = hi;
        while (l > low) {
            const double s = std::fabs(H(l - 1, l - 1)) + std::fabs(H(l, l));
            const double thresh = (s < 1e-300) ? eps * scale : eps * s;
            if (std::fabs(H(l, l - 1)) <= thresh) {
                H(l, l - 1) = 0.0;
                break;
            }
            --l;
        }

        if (l == hi) {
            eigenvalues.emplace_back(H(hi, hi), 0.0);
            --hi;
            iterations_since_deflation = 0;
            continue;
        }
        if (l == hi - 1) {
            std::complex<double> e1, e2;
            eig2x2(H(hi - 1, hi - 1), H(hi - 1, hi), H(hi, hi - 1), H(hi, hi), e1, e2);
            eigenvalues.push_back(e1);
            eigenvalues.push_back(e2);
            hi -= 2;
            iterations_since_deflation = 0;
            continue;
        }

        ++iterations_since_deflation;
        if (iterations_since_deflation > max_iterations) {
            for (long i = low; i <= hi; ++i) eigenvalues.emplace_back(H(i, i), 0.0 / 0.0);
            hi = low - 1;
            continue;
        }

        double shift_trace, shift_det;
        if (iterations_since_deflation % 20 == 0 && iterations_since_deflation > 0) {
            const double ad_hoc = std::fabs(H(hi, hi - 1)) + std::fabs(H(hi - 1, hi - 2));
            shift_trace = 2.0 * ad_hoc;
            shift_det = ad_hoc * ad_hoc;
        } else {
            const double a11 = H(hi - 1, hi - 1), a12 = H(hi - 1, hi);
            const double a21 = H(hi, hi - 1), a22 = H(hi, hi);
            shift_trace = a11 + a22;
            shift_det = a11 * a22 - a12 * a21;
        }

        const double h11 = H(l, l), h12 = H(l, l + 1);
        const double h21 = H(l + 1, l), h22 = H(l + 1, l + 1);
        const double h32 = (l + 2 <= hi) ? H(l + 2, l + 1) : 0.0;

        double x = h11 * h11 + h12 * h21 - shift_trace * h11 + shift_det;
        double y = h21 * (h11 + h22 - shift_trace);
        double z = h32 * h21;

        for (long k = l; k <= hi - 2; ++k) {
            const bool has_z = (k + 2 <= hi);
            const int len = has_z ? 3 : 2;
            double vec[3] = {x, y, z};
            Reflector h = make_householder(vec, len);

            const std::size_t col0 = static_cast<std::size_t>(std::max(l, k - 1));
            apply_left(H, h, static_cast<std::size_t>(k), col0, static_cast<std::size_t>(hi));
            const std::size_t row1 =
                static_cast<std::size_t>(std::min<long>(hi, k + 3));
            apply_right(H, h, static_cast<std::size_t>(l), row1, static_cast<std::size_t>(k));

            if (k < hi - 2) {
                x = H(k + 1, k);
                y = H(k + 2, k);
                z = (k + 3 <= hi) ? H(k + 3, k) : 0.0;
            }
        }

        {
            const long k = hi - 1;
            double vec[2] = {H(k, k - 1), H(k + 1, k - 1)};
            Reflector h = make_householder(vec, 2);
            const std::size_t col0 = static_cast<std::size_t>(std::max(l, k - 1));
            apply_left(H, h, static_cast<std::size_t>(k), col0, static_cast<std::size_t>(hi));
            apply_right(H, h, static_cast<std::size_t>(l), static_cast<std::size_t>(hi),
                        static_cast<std::size_t>(k));
        }
    }

    std::sort(eigenvalues.begin(), eigenvalues.end(),
              [](const std::complex<double>& a, const std::complex<double>& b) {
                  if (a.real() != b.real()) return a.real() < b.real();
                  return a.imag() < b.imag();
              });
    return eigenvalues;
}

}  // namespace f22
