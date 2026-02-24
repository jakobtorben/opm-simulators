/*
  Copyright 2026 SINTEF AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_DRAUGR_AMG_PRECONDITIONER_HEADER_INCLUDED
#define OPM_DRAUGR_AMG_PRECONDITIONER_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <draugr_amg.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm::Draugr
{

/**
 * @brief Wrapper for Draugr.jl AMG as a Dune preconditioner.
 *
 * This class provides an interface to the Draugr AMG library compiled from
 * Julia via PackageCompiler.jl. It wraps the C-callable API (draugr_amg_*)
 * into the Dune::PreconditionerWithUpdate interface used by OPM's linear
 * solvers.
 *
 * The preconditioner applies a single AMG cycle (V or W) and is intended for
 * use with scalar (1x1 block) matrices, e.g. as the pressure-solve component
 * in a CPR two-level scheme.
 *
 * Configuration is passed to Draugr as a JSON string. The OPM PropertyTree
 * (which already comes from JSON) is serialized back to a JSON string.
 *
 * Optimizations over a naive implementation:
 *   - Zero-copy apply(): passes raw Dune vector pointers directly to Julia,
 *     exploiting the fact that FieldVector<double,1> is layout-compatible
 *     with double and BlockVector stores elements contiguously.
 *   - Zero-copy values: nzvalPtr_() returns a direct pointer into the
 *     BCRSMatrix's contiguous value storage, avoiding any per-update copy
 *     of the nonzero values array.
 *   - Periodic full rebuild: like AmgxPreconditioner, periodically rebuilds
 *     the full AMG hierarchy to maintain coarsening quality as matrix
 *     coefficients evolve during the simulation.
 *
 * @tparam M The matrix type (BCRSMatrix with 1x1 FieldMatrix blocks)
 * @tparam X The domain (solution) vector type
 * @tparam Y The range (RHS) vector type
 * @tparam Comm The communication type (SequentialInformation for serial)
 */
template <class M, class X, class Y, class Comm>
class DraugrAMGPreconditioner : public Dune::PreconditionerWithUpdate<X, Y>
{
    // Ensure 1x1 blocks so that the contiguous memory optimization is valid
    static_assert(sizeof(typename M::block_type) == sizeof(double),
                  "DraugrAMGPreconditioner requires 1x1 scalar blocks");
    static_assert(sizeof(typename X::block_type) == sizeof(double),
                  "DraugrAMGPreconditioner requires 1x1 scalar block vectors");

public:
    using matrix_type = M;
    using domain_type = X;
    using range_type = Y;
    using field_type = typename X::field_type;

    /**
     * @brief Constructor.
     *
     * Initializes the Julia runtime (if not already done), creates an AMG
     * configuration from a JSON string built from the property tree, and
     * performs initial setup.
     *
     * @param A     The matrix for which the preconditioner is constructed.
     * @param prm   Property tree with configuration parameters.
     * @param comm  Parallel communicator (unused for now, serial only).
     */
    DraugrAMGPreconditioner(const M& A, const Opm::PropertyTree prm, const Comm& comm)
        : A_(A)
        , comm_(comm)
        , setup_frequency_(prm.get<int>("setup_frequency", 30))
        , update_counter_(0)
    {
        OPM_TIMEBLOCK(prec_construct);

        static std::once_flag julia_init_flag;
        std::call_once(julia_init_flag, []() { init_julia(0, nullptr); });

        // PropertyTree came from JSON — serialize it back and pass through.
        std::ostringstream oss;
        prm.write_json(oss, false);
        config_handle_ = draugr_amg_config_from_json(oss.str().c_str());

        if (config_handle_ < 0) {
            OPM_THROW(std::runtime_error,
                      "Draugr: config creation failed: "
                      + std::string(draugr_amg_last_error()));
        }

        extractCSR_();
        hierarchy_handle_ = draugr_amg_setup(
            static_cast<int32_t>(n_),
            static_cast<int32_t>(nnz_),
            rowptr_.data(),
            colval_.data(),
            nzvalPtr_(),
            config_handle_,
            static_cast<int32_t>(0),
            static_cast<int32_t>(1));

        if (hierarchy_handle_ < 0) {
            OPM_THROW(std::runtime_error,
                      "Draugr: setup failed: "
                      + std::string(draugr_amg_last_error()));
        }
    }

    ~DraugrAMGPreconditioner()
    {
        if (hierarchy_handle_ >= 0) {
            draugr_amg_free(hierarchy_handle_);
        }
        if (config_handle_ >= 0) {
            draugr_amg_config_free(config_handle_);
        }
    }

    // Non-copyable, non-movable (handles are not transferable)
    DraugrAMGPreconditioner(const DraugrAMGPreconditioner&) = delete;
    DraugrAMGPreconditioner& operator=(const DraugrAMGPreconditioner&) = delete;

    /**
     * @brief Update the preconditioner with current matrix values.
     *
     * Periodically rebuilds the full AMG hierarchy (every setup_frequency_
     * updates) to maintain coarsening quality as matrix coefficients evolve.
     * Between full rebuilds, uses draugr_amg_resetup() to update coefficients
     * while keeping the coarsening structure.
     *
     * No value buffers are copied on the C++ side: nzvalPtr_() passes a
     * direct pointer into BCRSMatrix's contiguous value storage.
     */
    void update() override
    {
        OPM_TIMEBLOCK(prec_update);

        ++update_counter_;
        const bool do_full_rebuild =
            (setup_frequency_ > 0) && ((update_counter_ % setup_frequency_) == 0);

        // partial=1: fast coefficient-only update (keeps coarsening structure)
        // partial=0: full rebuild reusing workspace arrays (re-coarsens)
        const int32_t partial = do_full_rebuild ? 0 : 1;
        const int32_t ret = draugr_amg_resetup(
            hierarchy_handle_,
            static_cast<int32_t>(n_),
            static_cast<int32_t>(nnz_),
            rowptr_.data(),
            colval_.data(),
            nzvalPtr_(),
            config_handle_,
            static_cast<int32_t>(0),
            partial,
            static_cast<int32_t>(1));

        if (ret < 0) {
            OPM_THROW(std::runtime_error,
                      "Draugr: resetup failed (partial=" + std::to_string(partial) + "): "
                      + std::string(draugr_amg_last_error()));
        }
    }

    /**
     * @brief Pre-processing step (no-op).
     */
    void pre(X& /*v*/, Y& /*d*/) override
    {
    }

    /**
     * @brief Apply one AMG cycle as preconditioner (zero-copy).
     *
     * Solves M*v = d approximately using a single AMG V/W-cycle,
     * where M approximates A. Passes raw Dune vector pointers directly
     * to the Julia library without intermediate buffer copies.
     *
     * This is safe because:
     *   - FieldVector<double,1> stores a single double (sizeof == sizeof(double))
     *   - BlockVector stores FieldVector elements contiguously in memory
     *   - amg_cycle!() never modifies the RHS vector b
     *
     * @param v  The update vector (output).
     * @param d  The defect/residual vector (input, not modified).
     */
    void apply(X& v, const Y& d) override
    {
        OPM_TIMEBLOCK(prec_apply);

        const int n = static_cast<int>(n_);

        // Zero the solution vector (preconditioner needs zero initial guess).
        // &v[0][0] gives a contiguous double* because FieldVector<double,1>
        // is layout-compatible with double.
        double* v_ptr = &(v[0][0]);
        std::fill(v_ptr, v_ptr + n, 0.0);

        // d is const; draugr_amg_cycle declares b as const double*.
        const double* d_ptr = &(d[0][0]);

        const int32_t ret = draugr_amg_cycle(
            hierarchy_handle_,
            static_cast<int32_t>(n),
            v_ptr,
            d_ptr,
            config_handle_);

        if (ret < 0) {
            OPM_THROW(std::runtime_error,
                      "Draugr: cycle failed: "
                      + std::string(draugr_amg_last_error()));
        }
    }

    /**
     * @brief Post-processing step (no-op).
     */
    void post(X& /*v*/) override
    {
    }

    /**
     * @brief Returns the solver category.
     */
    Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::sequential;
    }

    /**
     * @brief Indicates that the preconditioner can be updated in-place.
     *
     * Returns true because the periodic rebuild is handled internally.
     */
    bool hasPerfectUpdate() const override
    {
        return true;
    }

private:
    const M& A_;
    const Comm& comm_;

    int32_t config_handle_ = -1;
    int32_t hierarchy_handle_ = -1;

    // Periodic full-rebuild frequency (like AmgxPreconditioner)
    int setup_frequency_;
    int update_counter_;

    // CSR structure (0-based indexing); values read directly via nzvalPtr_()
    std::size_t n_ = 0;
    std::size_t nnz_ = 0;
    std::vector<int32_t> rowptr_;
    std::vector<int32_t> colval_;

    /**
     * @brief Extract CSR structure (rowptr, colval) from the DUNE BCRSMatrix.
     *
     * Values are not copied; nzvalPtr_() provides a direct pointer into
     * BCRSMatrix's contiguous storage.
     */
    void extractCSR_()
    {
        n_ = A_.N();
        nnz_ = A_.nonzeroes();

        rowptr_.resize(n_ + 1);
        colval_.resize(nnz_);

        int32_t pos = 0;
        rowptr_[0] = 0;
        for (auto row = A_.begin(); row != A_.end(); ++row) {
            for (auto col = row->begin(); col != row->end(); ++col) {
                colval_[pos] = static_cast<int32_t>(col.index());
                ++pos;
            }
            rowptr_[row.index() + 1] = pos;
        }
    }

    /**
     * @brief Direct pointer to BCRSMatrix's contiguous nonzero values.
     *
     * For 1x1 scalar blocks (FieldMatrix<double,1,1>), BCRSMatrix stores
     * all nonzero entries as contiguous doubles in CSR order.
     * This avoids any per-update copy of the values array.
     */
    const double* nzvalPtr_() const
    {
        return &((*A_.begin()->begin())[0][0]);
    }
};

} // namespace Opm::Draugr

#endif // OPM_DRAUGR_AMG_PRECONDITIONER_HEADER_INCLUDED
