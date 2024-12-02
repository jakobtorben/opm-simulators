#ifndef OPM_AMGX_PRECONDITIONER_HEADER_INCLUDED
#define OPM_AMGX_PRECONDITIONER_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <amgx_c.h>

#include <memory>
#include <vector>

namespace Amgx {


struct AmgxConfig {
    std::string solver = "AMG";
    std::string algorithm = "CLASSICAL";
    std::string interpolator = "D2";
    std::string selector = "PMIS";
    int presweeps = 3;
    int postsweeps = 3;
    double strength_threshold = 0.5;
    int max_iters = 1;

    std::string toString() const {
        return "config_version=2, "
               "solver=" + solver + ", "
               "algorithm=" + algorithm + ", "
               "interpolator=" + interpolator + ", "
               "selector=" + selector + ", "
               "presweeps=" + std::to_string(presweeps) + ", "
               "postsweeps=" + std::to_string(postsweeps) + ", "
               "strength_threshold=" + std::to_string(strength_threshold) + ", "
               "max_iters=" + std::to_string(max_iters);
    }
};


template<class Matrix, class Vector>
class AmgxPreconditioner : public Dune::PreconditionerWithUpdate<Vector,Vector>
{
public:
    using matrix_type = Matrix;
    using domain_type = Vector;
    using range_type = Vector;
    using field_type = typename Vector::field_type;

    static constexpr int block_size = 1;

    AmgxPreconditioner(const Matrix& A, const Opm::PropertyTree& prm)
        : matrix_(A)
        , prm_(prm)
        , N_(matrix_.N())
        , nnz_(matrix_.nonzeroes())
        , row_ptrs_(N_ + 1)
        , col_indices_(nnz_)
    {
        OPM_TIMEBLOCK(prec_construct);

        // Create configuration
        AMGX_SAFE_CALL(AMGX_config_create(&cfg_, config_.toString().c_str()));
        AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc_, cfg_));

        // Create solver and matrix/vector handles
        AMGX_SAFE_CALL(AMGX_solver_create(&solver_, rsrc_, AMGX_mode_dDDI, cfg_));
        AMGX_SAFE_CALL(AMGX_matrix_create(&A_amgx_, rsrc_, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&x_amgx_, rsrc_, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&b_amgx_, rsrc_, AMGX_mode_dDDI));

        // Setup matrix structure
        setupSparsityPattern();

        // initialize matrix with values
        const double* values = &(matrix_[0][0][0][0]);
        AMGX_SAFE_CALL(AMGX_pin_memory(const_cast<double*>(values), sizeof(double) * nnz_ * block_size * block_size));
        AMGX_SAFE_CALL(AMGX_matrix_upload_all(A_amgx_, N_, nnz_, block_size, block_size,
                                             row_ptrs_.data(), col_indices_.data(),
                                             values, nullptr));
        update();
    }

    ~AmgxPreconditioner()
    {
        const double* values = &(matrix_[0][0][0][0]);
        AMGX_SAFE_CALL(AMGX_unpin_memory(const_cast<double*>(values)));
        if (solver_) {
            AMGX_SAFE_CALL(AMGX_solver_destroy(solver_));
        }
        if (x_amgx_) {
            AMGX_SAFE_CALL(AMGX_vector_destroy(x_amgx_));
        }
        if (b_amgx_) {
            AMGX_SAFE_CALL(AMGX_vector_destroy(b_amgx_));
        }
        if (A_amgx_) {
            AMGX_SAFE_CALL(AMGX_matrix_destroy(A_amgx_));
        }
        // Destroying resrouces and config crashes when reinitializing
        //if (rsrc_) {
        //    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc_));
        //}
        //if (cfg_) {
        //    AMGX_SAFE_CALL(AMGX_config_destroy(cfg_));
        //}
    }

    void pre(Vector& x, Vector& b) override
    {
        DUNE_UNUSED_PARAMETER(x);
        DUNE_UNUSED_PARAMETER(b);
    }

    void apply(Vector& v, const Vector& d) override
    {
        OPM_TIMEBLOCK(prec_apply);

        // Upload vectors to AMGX
        AMGX_SAFE_CALL(AMGX_vector_upload(x_amgx_, N_, block_size, &v[0][0]));
        AMGX_SAFE_CALL(AMGX_vector_upload(b_amgx_, N_, block_size, &d[0][0]));

        // Apply preconditioner
        AMGX_SAFE_CALL(AMGX_solver_solve(solver_, b_amgx_, x_amgx_));

        // Download result
        AMGX_SAFE_CALL(AMGX_vector_download(x_amgx_, &v[0][0]));
    }

    void post(Vector& x) override
    {
        DUNE_UNUSED_PARAMETER(x);
    }

    void update() override
    {
        OPM_TIMEBLOCK(prec_update);
        copyMatrixToAmgx();

        if (update_counter_ == 0) {
            AMGX_SAFE_CALL(AMGX_solver_setup(solver_, A_amgx_));
        } else {
            AMGX_SAFE_CALL(AMGX_solver_resetup(solver_, A_amgx_));
        }

        ++update_counter_;
        if (update_counter_ >= setup_frequency_) {
            update_counter_ = 0;
        }
    }

    Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::sequential;
    }

    bool hasPerfectUpdate() const override
    {
        // The Amgx preconditioner can depend on the values of the matrix, so it must be recreated
        return false;
    }

private:
    static constexpr int setup_frequency_ = 30;
    int update_counter_ = 0;

    void setupSparsityPattern()
    {
        int pos = 0;
        row_ptrs_[0] = 0;
        for (auto row = matrix_.begin(); row != matrix_.end(); ++row) {
            for (auto col = row->begin(); col != row->end(); ++col) {
                col_indices_[pos++] = col.index();
            }
            row_ptrs_[row.index() + 1] = pos;
        }
    }

    void copyMatrixToAmgx()
    {
        // Get direct pointer to matrix values
        const double* values = &(matrix_[0][0][0][0]);
        // update matrix with new values, assuming the sparsity structure is the same
        AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A_amgx_, N_, nnz_, values, nullptr));
    }

    const Matrix& matrix_;
    const Opm::PropertyTree& prm_;
    const int N_;
    const int nnz_;
    const AmgxConfig config_;

    AMGX_config_handle cfg_ = nullptr;
    AMGX_resources_handle rsrc_ = nullptr;
    AMGX_solver_handle solver_ = nullptr;
    AMGX_matrix_handle A_amgx_ = nullptr;
    AMGX_vector_handle x_amgx_ = nullptr;
    AMGX_vector_handle b_amgx_ = nullptr;

    std::vector<int> row_ptrs_;
    std::vector<int> col_indices_;
};

} // namespace Amgx

#endif // OPM_AMGX_PRECONDITIONER_HEADER_INCLUDED
