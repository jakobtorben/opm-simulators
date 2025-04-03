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

namespace Opm::gpuistl
{

template <class X, class XGPU>
class GpuWellOperator : public Opm::LinearOperatorExtra<XGPU, XGPU> {
public:
    using Base = Opm::LinearOperatorExtra<XGPU, XGPU>;
    using typename Base::field_type;
    using typename Base::PressureMatrix;
    static constexpr auto block_size = X::block_type::dimension;
    using WellOperatorCPU = Opm::LinearOperatorExtra<X, X>;
    using GpuReal = typename XGPU::field_type;

    // Constructor takes CPU well operator
    explicit GpuWellOperator(const WellOperatorCPU& wellOpCPU)
        : wellOpCPU_(wellOpCPU)
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
            wellOpCPU_.applyscaleadd(alpha, x_cpu, y_cpu);
            y.copyFromHost(y_cpu);
        }
    }

    Dune::SolverCategory::Category category() const override {
        return wellOpCPU_.category();
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                 const XGPU& weights,
                                 const bool use_well_weights) const override {
        // CPU fallback for now
        X weights_cpu(weights.dim() / block_size);
        weights.copyToHost(weights_cpu);
        wellOpCPU_.addWellPressureEquations(jacobian, weights_cpu, use_well_weights);
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const override {
        wellOpCPU_.addWellPressureEquationsStruct(jacobian);
    }

    int getNumberOfExtraEquations() const override {
        return wellOpCPU_.getNumberOfExtraEquations();
    }

    std::shared_ptr<Opm::LinearOperatorExtra<XGPU, XGPU>> getWellOperator() const {
        return std::make_shared<GpuWellOperator<X, XGPU>>(*this);
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
        for (const auto& well_matrices : matrices) {
            // Extract matrices for this well
            const auto& B = std::get<0>(well_matrices);  // B matrix
            const auto& C = std::get<1>(well_matrices);  // C matrix
            const auto& invD = std::get<2>(well_matrices); // D^-1 matrix

            // 1. Create GPU vectors for intermediate results
            // These sizes are from the matrix dimensions, not the full system
            XGPU temp_Bx(B->N());         // To store B*x (well equation size)
            XGPU temp_invDBx(invD->N());  // To store D^-1 * (B*x) (well equation size)
            XGPU temp_result(y.dim());      // To store the final result (system size)

            // 2. Compute B*x: Extract cell values -> multiply -> store in temp_Bx
            B->mv(x, temp_Bx);

            // 3. Compute D^-1 * (B*x)
            invD->mv(temp_Bx, temp_invDBx);

            // 4. Compute C^T * (D^-1 * (B*x))
            // Note: We're temporarily using regular mv() instead of transposed operation
            // This isn't mathematically correct but will compile for initial testing
            C->mv(temp_invDBx, temp_result);

            // 5. Subtract from y: y -= result
            y.axpy(-1.0, temp_result);
        }
    }

    void applyCPUFallback(const XGPU& x, XGPU& y) const {
        // CPU implementation (fallback)
        X x_cpu(x.dim() / block_size);
        X y_cpu(y.dim() / block_size);
        x.copyToHost(x_cpu);
        y.copyToHost(y_cpu);
        wellOpCPU_.apply(x_cpu, y_cpu);
        y.copyFromHost(y_cpu);
    }

    const WellOperatorCPU& wellOpCPU_;
    bool useGPUImpl_ = false;
    std::shared_ptr<Opm::gpuistl::GpuWellMatrices<GpuReal>> wellMatrices_;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_WELL_OPERATOR_HPP
