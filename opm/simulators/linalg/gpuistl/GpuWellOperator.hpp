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
#include <vector>
#include <iostream>
#include <typeinfo>

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
        // Default to CPU fallback for now
        // When a real GPU implementation is available, this can be enabled
        useGPUImpl_ = false;
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

        // If we have matrices, use the GPU implementation
        useGPUImpl_ = wellMatrices_ && !wellMatrices_->empty();
    }

    // No-op implementation of the pure virtual method from LinearOperatorExtra
    void setWellMatrices(const std::shared_ptr<Opm::SetableWellOperatorBase>& wellOperator) const override {
    }

    void apply(const XGPU& x, XGPU& y) const override {
        if (useGPUImpl_) {
            applyGPU(x, y);
        } else {
            applyCPUFallback(x, y);
        }
    }

    void applyscaleadd(field_type alpha, const XGPU& x, XGPU& y) const override {
        if (useGPUImpl_) {
            // Apply on GPU with scaling
            XGPU temp(y.dim());
            applyGPU(x, temp);
            y.axpy(alpha, temp); // y += alpha * temp
        } else {
            // CPU fallback
            X x_cpu(x.dim() / block_size);
            X y_cpu(y.dim() / block_size);
            x.copyToHost(x_cpu);
            y.copyToHost(y_cpu);
            wellOpCPU_ptr_->applyscaleadd(alpha, x_cpu, y_cpu);
            y.copyFromHost(y_cpu);
        }
    }

    Dune::SolverCategory::Category category() const override {
        return wellOpCPU_ptr_->category();
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
    void applyGPU(const XGPU& x, XGPU& y) const {
        // Implementation of y -= (C^T * (D^-1 * (B*x))) on GPU
        if (!wellMatrices_ || wellMatrices_->empty()) {
            // No matrices available, do nothing
            return;
        }

        // For each well in the matrices:
        const auto& matrices = wellMatrices_->getMatrices();

        for (size_t i = 0; i < matrices.size(); ++i) {
            const auto& well_matrices = matrices[i];
            // Extract matrices for this well
            const auto& B = std::get<0>(well_matrices);  // B matrix
            const auto& C = std::get<1>(well_matrices);  // C matrix
            const auto& invD = std::get<2>(well_matrices); // D^-1 matrix
            // Get the cell indices for this well
            const auto& cellIndices = wellMatrices_->getWellCellIndices(i);

            // Create subset vectors for the perforated cells
            XGPU x_local = x.createSubset(cellIndices, block_size);
            XGPU result_local(cellIndices.dim() * block_size);

            // Define the number of static well equations based on the matrix dimensions
            const int numStaticWellEq = B->N();  // Get the actual number of rows in B

            // Note that GpuSparseMatrix used here is not really appropriate here, given that it:
            // 1. does not support non-square blocks.
            // 2. does not support non-square matrices at all.
            // 3. does not support block size = 1.
            // In the current implementation, B, C, and invD are implemented
            // as non-blocked matrices, but given the constraints above this is not expected to work yet.
            // In the local representation of the well equations, the matrices are actually not
            // sparse, so we should probably not use the GpuSparseMatrix class.

            // Matrix dimensions:
            // B matrix: NNZ = numPerforations, block_size = (numStaticWellEq x numEq)
            // C^T matrix: NNZ = numPerforations, block_size = (numStaticWellEq x numEq)
            // invD matrix: NNZ = 1, block_size = (numStaticWellEq x numStaticWellEq)

            // Draft implementation to test the approach:
            // Needs to be fixed in the final implementation, to do the correct operations,
            // check the dimensions, and implement the inverse of D solve.

            // 1. Compute B*x_local (well equations applied to perforated cells)
            XGPU temp_Bx(numStaticWellEq);
            B->mv(x_local, temp_Bx);

            // 2. Compute D^-1 * (B*x)
            XGPU temp_invDBx(numStaticWellEq);
            invD->mv(temp_Bx, temp_invDBx);

            // 3. For C^T * (D^-1 * (B*x))
            C->mv(temp_invDBx, result_local);

            // 4. Subtract from y only at the perforated cell locations
            y.axpy(-1.0, result_local);
        }
    }

    void applyCPUFallback(const XGPU& x, XGPU& y) const {
        // CPU implementation (fallback)
        X x_cpu(x.dim() / block_size);
        X y_cpu(y.dim() / block_size);
        x.copyToHost(x_cpu);
        y.copyToHost(y_cpu);
        wellOpCPU_ptr_->apply(x_cpu, y_cpu);
        y.copyFromHost(y_cpu);
    }

    const std::shared_ptr<const WellOperatorCPU> wellOpCPU_ptr_;
    bool useGPUImpl_ = false;
    std::shared_ptr<Opm::gpuistl::GpuWellMatrices<GpuReal>> wellMatrices_;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_WELL_OPERATOR_HPP
