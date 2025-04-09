/*
  Copyright 2022-2024 SINTEF AS
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

#ifndef OPM_DILU_HEADER_INCLUDED
#define OPM_DILU_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/common/unused.hh>
#include <dune/common/version.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <cstddef>
#include <vector>

namespace Dune
{

/*! \brief The DILU (Diagonal Incomplete LU) preconditioner.
 *  \details Optimized sequential implementation without multithreading overhead.
             Use MultithreadDILU for parallel execution.

   \tparam M The matrix type to operate on
   \tparam X Type of the update
   \tparam Y Type of the defect
*/
template <class M, class X, class Y>
class DILU : public PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    using matrix_type = M;
    //! \brief The domain type of the preconditioner.
    using domain_type = X;
    //! \brief The range type of the preconditioner.
    using range_type = Y;
    //! \brief The field type of the preconditioner.
    using field_type = typename X::field_type;
    //! \brief scalar type underlying the field_type

    /*! \brief Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
    */
    explicit DILU(const M& A)
        : A_(A)
    {
        OPM_TIMEBLOCK(prec_construct);
        Dinv_.resize(A_.N());
        update();
    }

    /*!
       \brief Update the preconditioner.
       \copydoc Preconditioner::update()
    */
    void update() override
    {
        OPM_TIMEBLOCK(prec_update);
        serialUpdate();
    }

    /*!
       \brief Prepare the preconditioner.
       \copydoc Preconditioner::pre(X&,Y&)
    */
    void pre(X& v, Y& d) override
    {
        DUNE_UNUSED_PARAMETER(v);
        DUNE_UNUSED_PARAMETER(d);
    }


    /*!
       \brief Apply the preconditioner.
       \copydoc Preconditioner::apply(X&,const Y&)
    */
    void apply(X& v, const Y& d) override
    {
        OPM_TIMEBLOCK(prec_apply);
        serialApply(v, d);
    }

    /*!
       \brief Clean up.
       \copydoc Preconditioner::post(X&)
    */
    void post(X& x) override
    {
        DUNE_UNUSED_PARAMETER(x);
    }

    std::vector<typename M::block_type> getDiagonal()
    {
        return Dinv_;
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const override
    {
        return SolverCategory::sequential;
    }

    virtual bool hasPerfectUpdate() const override {
        return true;
    }

private:
    //! \brief The matrix we operate on.
    const M& A_;
    //! \brief The inverse of the diagnal matrix
    std::vector<typename M::block_type> Dinv_;

    void serialUpdate()
    {
        for (std::size_t row = 0; row < A_.N(); ++row) {
            Dinv_[row] = A_[row][row];
        }
        for (auto row = A_.begin(); row != A_.end(); ++row) {
            const auto row_i = row.index();
            auto Dinv_temp = Dinv_[row_i];
            for (auto a_ij = row->begin(); a_ij.index() < row_i; ++a_ij) {
                const auto col_j = a_ij.index();
                const auto a_ji = A_[col_j].find(row_i);
                // if A[i, j] != 0 and A[j, i] != 0
                if (a_ji != A_[col_j].end()) {
                    // Dinv_temp -= A[i, j] * d[j] * A[j, i]
                    Dinv_temp -= (*a_ij) * Dune::FieldMatrix(Dinv_[col_j]) * (*a_ji);
                }
            }
            Dinv_temp.invert();
            Dinv_[row_i] = Dinv_temp;
        }
    }

    void serialApply(X& v, const Y& d)
    {
        // M = (D + L_A) D^-1 (D + U_A)   (a LU decomposition of M)
        // where L_A and U_A are the strictly lower and upper parts of A and M has the properties:
        // diag(A) = diag(M)
        // Working with defect d = b - Ax and update v = x_{n+1} - x_n
        // solving the product M^-1(d) using upper and lower triangular solve
        // v = M^{-1}*d = (D + U_A)^{-1} D (D + L_A)^{-1} * d
        // lower triangular solve: (D + L_A) y = d
        using Xblock = typename X::block_type;
        using Yblock = typename Y::block_type;
        {
            OPM_TIMEBLOCK(lower_solve);
            auto endi = A_.end();
            for (auto row = A_.begin(); row != endi; ++row) {
                const auto row_i = row.index();
                Yblock rhs = d[row_i];
                for (auto a_ij = (*row).begin(); a_ij.index() < row_i; ++a_ij) {
                    // if  A[i][j] != 0
                    // rhs -= A[i][j]* y[j], where v_j stores y_j
                    const auto col_j = a_ij.index();
                    a_ij->mmv(v[col_j], rhs);
                }
                // y_i = Dinv_i * rhs
                // storing y_i in v_i
                Dinv_[row_i].mv(rhs, v[row_i]); // (D + L_A)_ii = D_i
            }
        }

        {
            OPM_TIMEBLOCK(upper_solve);

            // upper triangular solve: (D + U_A) v = Dy
            auto rendi = A_.beforeBegin();
            for (auto row = A_.beforeEnd(); row != rendi; --row) {
                const auto row_i = row.index();
                // rhs = 0
                Xblock rhs(0.0);
                for (auto a_ij = (*row).beforeEnd(); a_ij.index() > row_i; --a_ij) {
                    // if A[i][j] != 0
                    // rhs += A[i][j]*v[j]
                    const auto col_j = a_ij.index();
                    a_ij->umv(v[col_j], rhs);
                }
                // calculate update v = M^-1*d
                // v_i = y_i - Dinv_i*rhs
                // before update v_i is y_i
                Dinv_[row_i].mmv(rhs, v[row_i]);
            }
        }
    }
};

} // namespace Dune

#endif
