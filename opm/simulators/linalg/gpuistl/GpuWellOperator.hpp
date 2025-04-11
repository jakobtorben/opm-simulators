/*
  Copyright 2025 Equinor ASA.

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

#ifndef OPM_GPU_WELL_OPERATOR_HPP
#define OPM_GPU_WELL_OPERATOR_HPP

#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/GpuWellMatrices.hpp>
#include <opm/simulators/wells/WellInterface.hpp>

#include <memory>

namespace Opm::gpuistl {
class GpuWellMatricesBase;
template <class Scalar> class GpuWellMatrices;
}

namespace Opm::gpuistl
{

template <class X, class XGPU>
class GpuWellOperator : public Opm::LinearOperatorExtra<XGPU, XGPU>,
                        public Opm::SetableWellOperator<XGPU>,
                        public Opm::SetableWellOperatorBase
{
public:
    using Base = Opm::LinearOperatorExtra<XGPU, XGPU>;
    using typename Base::field_type;
    using typename Base::PressureMatrix;
    static constexpr auto block_size = X::block_type::dimension;
    using WellOperatorCPU = Opm::LinearOperatorExtra<X, X>;
    using GpuReal = typename XGPU::field_type;

    // Constructor takes CPU well operator
    explicit GpuWellOperator(const std::shared_ptr<const WellOperatorCPU>& wellOpCPU)
        : wellOpCPU_ptr_(wellOpCPU)
    {
    }

    /**
     * @brief Set GPU matrices from a GpuWellMatrices object
     *
     * This method configures the well operator to use precomputed GPU matrices
     * from the GpuWellMatrices storage object.
     *
     * @param wellMatrices Shared pointer to the well matrices storage
     */
    void setWellMatrices(const std::shared_ptr<Opm::gpuistl::GpuWellMatrices<GpuReal>>& wellMatrices) override
    {
        // Store the reference to the matrices storage
        wellMatrices_ = wellMatrices;
    }

    // No-op implementation of the pure virtual method from LinearOperatorExtra
    void setWellMatrices(const std::shared_ptr<Opm::SetableWellOperatorBase>& wellOperator) const override {
    }

    void apply(const XGPU& x, XGPU& Ax) const {
        // Implementation of y -= (C^T * (D^-1 * (B*x))) on GPU
        if (!wellMatrices_ || wellMatrices_->empty()) {
            // No matrices available, do nothing
            return;
        }

        // For each well in the matrices:
        const auto& matrices = wellMatrices_->getMatrices();

        for (size_t i = 0; i < matrices.size(); ++i) {

            if (wellMatrices_->shouldSkip(i)) {
                continue;
            }

            const auto& well_matrices = matrices[i];
            // Extract matrices for this well
            const auto& B = std::get<0>(well_matrices);  // B matrix
            const auto& C = std::get<1>(well_matrices);  // C matrix
            const auto& invD = std::get<2>(well_matrices); // D^-1 matrix
            // Get the cell indices for this well
            const auto& cellIndices = wellMatrices_->getWellCellIndices(i);

            // Define the number of perforations and static well equations based on the matrix dimensions
            const auto numPerfs = cellIndices.dim();
            const auto numStaticWellEq = B->rows();

            // TODO: [perf] Don't reallocate the vectors for every apply

            // Create subset vectors for the perforated cells
            XGPU x_local = x.createSubset(cellIndices, block_size);
            XGPU Ax_local = Ax.createSubset(cellIndices, block_size);

            // Compute Ax -= C^T * (D^-1 * (B*x)) for this well:
            // 1. Compute B*x_local
            XGPU Bx(numStaticWellEq);
            B->mv(x_local, Bx);

            // 2. Compute D^-1 * (B*x)
            // This applies the inverted well diagonal matrix
            XGPU invDBx(numStaticWellEq);
            invD->mv(Bx, invDBx);

            // Ax = Ax - duneC_^T * invDBx
            C->mmtv(invDBx, Ax_local);

            // Write Ax_local to Ax at the perforation locations
            Ax_local.writeSubsetBack(Ax, cellIndices, block_size);
        }
    }

    void applyscaleadd(field_type alpha, const XGPU& x, XGPU& y) const override {
        XGPU scaleAddRes_(y.dim());
        scaleAddRes_ = 0.0;
        // scaleAddRes_  = - C D^-1 B x
        apply(x, scaleAddRes_);
        // Ax = Ax + alpha * scaleAddRes_
        y.axpy(alpha, scaleAddRes_);
    }

    Dune::SolverCategory::Category category() const override {
        return Dune::SolverCategory::sequential;
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                 const XGPU& weights,
                                 const bool use_well_weights) const override {
        // CPU fallback for now
        X weights_cpu(weights.dim() / block_size);
        weights.copyToHost(weights_cpu);
        wellOpCPU_ptr_->addWellPressureEquations(jacobian, weights_cpu, use_well_weights);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const override {
        wellOpCPU_ptr_->addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const override {
        return wellOpCPU_ptr_->getNumberOfExtraEquations();
    }

    std::shared_ptr<Opm::LinearOperatorExtra<XGPU, XGPU>> getWellOperator() const {
        return std::make_shared<GpuWellOperator<X, XGPU>>(*this);
    }

    std::shared_ptr<Opm::gpuistl::GpuWellMatrices<GpuReal>> getWellMatrices() const {
        return wellMatrices_;
    }

    void setWellMatricesImpl(const std::shared_ptr<Opm::gpuistl::GpuWellMatricesBase>& baseMatrices) override {
        // Cast and use the matrices
        auto typedMatrices = std::static_pointer_cast<Opm::gpuistl::GpuWellMatrices<GpuReal>>(baseMatrices);
        this->setWellMatrices(typedMatrices);
    }

private:


    const std::shared_ptr<const WellOperatorCPU> wellOpCPU_ptr_;
    std::shared_ptr<Opm::gpuistl::GpuWellMatrices<GpuReal>> wellMatrices_;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_WELL_OPERATOR_HPP
