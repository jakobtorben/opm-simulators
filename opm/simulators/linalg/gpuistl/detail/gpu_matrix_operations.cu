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

#include <opm/simulators/linalg/gpuistl/detail/gpu_matrix_operations.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpuThreadUtils.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace Opm::gpuistl::detail
{
namespace
{
    // CUDA kernel to invert a 4x4 matrix in column-major format
    template <class T>
    __global__ void cuInvert4x4ColMajor(const T* matrix, T* inverse, int* success)
    {
        // Only need a single thread to do this work since we're operating on a single small matrix
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Helper lambda to access elements in column-major format
            // Element (i,j) is at index j*4 + i in column-major format
            auto el = [](const T* mat, int i, int j) -> T { return mat[j*4 + i]; };

            // Calculate cofactors for each element of the inverse matrix

            // First column of inverse
            inverse[0*4 + 0] =  el(matrix, 1, 1) * el(matrix, 2, 2) * el(matrix, 3, 3) -
                                el(matrix, 1, 1) * el(matrix, 2, 3) * el(matrix, 3, 2) -
                                el(matrix, 2, 1) * el(matrix, 1, 2) * el(matrix, 3, 3) +
                                el(matrix, 2, 1) * el(matrix, 1, 3) * el(matrix, 3, 2) +
                                el(matrix, 3, 1) * el(matrix, 1, 2) * el(matrix, 2, 3) -
                                el(matrix, 3, 1) * el(matrix, 1, 3) * el(matrix, 2, 2);

            inverse[0*4 + 1] = -el(matrix, 0, 1) * el(matrix, 2, 2) * el(matrix, 3, 3) +
                                el(matrix, 0, 1) * el(matrix, 2, 3) * el(matrix, 3, 2) +
                                el(matrix, 2, 1) * el(matrix, 0, 2) * el(matrix, 3, 3) -
                                el(matrix, 2, 1) * el(matrix, 0, 3) * el(matrix, 3, 2) -
                                el(matrix, 3, 1) * el(matrix, 0, 2) * el(matrix, 2, 3) +
                                el(matrix, 3, 1) * el(matrix, 0, 3) * el(matrix, 2, 2);

            inverse[0*4 + 2] =  el(matrix, 0, 1) * el(matrix, 1, 2) * el(matrix, 3, 3) -
                                el(matrix, 0, 1) * el(matrix, 1, 3) * el(matrix, 3, 2) -
                                el(matrix, 1, 1) * el(matrix, 0, 2) * el(matrix, 3, 3) +
                                el(matrix, 1, 1) * el(matrix, 0, 3) * el(matrix, 3, 2) +
                                el(matrix, 3, 1) * el(matrix, 0, 2) * el(matrix, 1, 3) -
                                el(matrix, 3, 1) * el(matrix, 0, 3) * el(matrix, 1, 2);

            inverse[0*4 + 3] = -el(matrix, 0, 1) * el(matrix, 1, 2) * el(matrix, 2, 3) +
                                el(matrix, 0, 1) * el(matrix, 1, 3) * el(matrix, 2, 2) +
                                el(matrix, 1, 1) * el(matrix, 0, 2) * el(matrix, 2, 3) -
                                el(matrix, 1, 1) * el(matrix, 0, 3) * el(matrix, 2, 2) -
                                el(matrix, 2, 1) * el(matrix, 0, 2) * el(matrix, 1, 3) +
                                el(matrix, 2, 1) * el(matrix, 0, 3) * el(matrix, 1, 2);

            // Second column of inverse
            inverse[1*4 + 0] = -el(matrix, 1, 0) * el(matrix, 2, 2) * el(matrix, 3, 3) +
                                el(matrix, 1, 0) * el(matrix, 2, 3) * el(matrix, 3, 2) +
                                el(matrix, 2, 0) * el(matrix, 1, 2) * el(matrix, 3, 3) -
                                el(matrix, 2, 0) * el(matrix, 1, 3) * el(matrix, 3, 2) -
                                el(matrix, 3, 0) * el(matrix, 1, 2) * el(matrix, 2, 3) +
                                el(matrix, 3, 0) * el(matrix, 1, 3) * el(matrix, 2, 2);

            inverse[1*4 + 1] =  el(matrix, 0, 0) * el(matrix, 2, 2) * el(matrix, 3, 3) -
                                el(matrix, 0, 0) * el(matrix, 2, 3) * el(matrix, 3, 2) -
                                el(matrix, 2, 0) * el(matrix, 0, 2) * el(matrix, 3, 3) +
                                el(matrix, 2, 0) * el(matrix, 0, 3) * el(matrix, 3, 2) +
                                el(matrix, 3, 0) * el(matrix, 0, 2) * el(matrix, 2, 3) -
                                el(matrix, 3, 0) * el(matrix, 0, 3) * el(matrix, 2, 2);

            inverse[1*4 + 2] = -el(matrix, 0, 0) * el(matrix, 1, 2) * el(matrix, 3, 3) +
                                el(matrix, 0, 0) * el(matrix, 1, 3) * el(matrix, 3, 2) +
                                el(matrix, 1, 0) * el(matrix, 0, 2) * el(matrix, 3, 3) -
                                el(matrix, 1, 0) * el(matrix, 0, 3) * el(matrix, 3, 2) -
                                el(matrix, 3, 0) * el(matrix, 0, 2) * el(matrix, 1, 3) +
                                el(matrix, 3, 0) * el(matrix, 0, 3) * el(matrix, 1, 2);

            inverse[1*4 + 3] =  el(matrix, 0, 0) * el(matrix, 1, 2) * el(matrix, 2, 3) -
                                el(matrix, 0, 0) * el(matrix, 1, 3) * el(matrix, 2, 2) -
                                el(matrix, 1, 0) * el(matrix, 0, 2) * el(matrix, 2, 3) +
                                el(matrix, 1, 0) * el(matrix, 0, 3) * el(matrix, 2, 2) +
                                el(matrix, 2, 0) * el(matrix, 0, 2) * el(matrix, 1, 3) -
                                el(matrix, 2, 0) * el(matrix, 0, 3) * el(matrix, 1, 2);

            // Third column of inverse
            inverse[2*4 + 0] =  el(matrix, 1, 0) * el(matrix, 2, 1) * el(matrix, 3, 3) -
                                el(matrix, 1, 0) * el(matrix, 2, 3) * el(matrix, 3, 1) -
                                el(matrix, 2, 0) * el(matrix, 1, 1) * el(matrix, 3, 3) +
                                el(matrix, 2, 0) * el(matrix, 1, 3) * el(matrix, 3, 1) +
                                el(matrix, 3, 0) * el(matrix, 1, 1) * el(matrix, 2, 3) -
                                el(matrix, 3, 0) * el(matrix, 1, 3) * el(matrix, 2, 1);

            inverse[2*4 + 1] = -el(matrix, 0, 0) * el(matrix, 2, 1) * el(matrix, 3, 3) +
                                el(matrix, 0, 0) * el(matrix, 2, 3) * el(matrix, 3, 1) +
                                el(matrix, 2, 0) * el(matrix, 0, 1) * el(matrix, 3, 3) -
                                el(matrix, 2, 0) * el(matrix, 0, 3) * el(matrix, 3, 1) -
                                el(matrix, 3, 0) * el(matrix, 0, 1) * el(matrix, 2, 3) +
                                el(matrix, 3, 0) * el(matrix, 0, 3) * el(matrix, 2, 1);

            inverse[2*4 + 2] =  el(matrix, 0, 0) * el(matrix, 1, 1) * el(matrix, 3, 3) -
                                el(matrix, 0, 0) * el(matrix, 1, 3) * el(matrix, 3, 1) -
                                el(matrix, 1, 0) * el(matrix, 0, 1) * el(matrix, 3, 3) +
                                el(matrix, 1, 0) * el(matrix, 0, 3) * el(matrix, 3, 1) +
                                el(matrix, 3, 0) * el(matrix, 0, 1) * el(matrix, 1, 3) -
                                el(matrix, 3, 0) * el(matrix, 0, 3) * el(matrix, 1, 1);

            inverse[2*4 + 3] = -el(matrix, 0, 0) * el(matrix, 1, 1) * el(matrix, 2, 3) +
                                el(matrix, 0, 0) * el(matrix, 1, 3) * el(matrix, 2, 1) +
                                el(matrix, 1, 0) * el(matrix, 0, 1) * el(matrix, 2, 3) -
                                el(matrix, 1, 0) * el(matrix, 0, 3) * el(matrix, 2, 1) -
                                el(matrix, 2, 0) * el(matrix, 0, 1) * el(matrix, 1, 3) +
                                el(matrix, 2, 0) * el(matrix, 0, 3) * el(matrix, 1, 1);

            // Fourth column of inverse
            inverse[3*4 + 0] = -el(matrix, 1, 0) * el(matrix, 2, 1) * el(matrix, 3, 2) +
                                el(matrix, 1, 0) * el(matrix, 2, 2) * el(matrix, 3, 1) +
                                el(matrix, 2, 0) * el(matrix, 1, 1) * el(matrix, 3, 2) -
                                el(matrix, 2, 0) * el(matrix, 1, 2) * el(matrix, 3, 1) -
                                el(matrix, 3, 0) * el(matrix, 1, 1) * el(matrix, 2, 2) +
                                el(matrix, 3, 0) * el(matrix, 1, 2) * el(matrix, 2, 1);

            inverse[3*4 + 1] =  el(matrix, 0, 0) * el(matrix, 2, 1) * el(matrix, 3, 2) -
                                el(matrix, 0, 0) * el(matrix, 2, 2) * el(matrix, 3, 1) -
                                el(matrix, 2, 0) * el(matrix, 0, 1) * el(matrix, 3, 2) +
                                el(matrix, 2, 0) * el(matrix, 0, 2) * el(matrix, 3, 1) +
                                el(matrix, 3, 0) * el(matrix, 0, 1) * el(matrix, 2, 2) -
                                el(matrix, 3, 0) * el(matrix, 0, 2) * el(matrix, 2, 1);

            inverse[3*4 + 2] = -el(matrix, 0, 0) * el(matrix, 1, 1) * el(matrix, 3, 2) +
                                el(matrix, 0, 0) * el(matrix, 1, 2) * el(matrix, 3, 1) +
                                el(matrix, 1, 0) * el(matrix, 0, 1) * el(matrix, 3, 2) -
                                el(matrix, 1, 0) * el(matrix, 0, 2) * el(matrix, 3, 1) -
                                el(matrix, 3, 0) * el(matrix, 0, 1) * el(matrix, 1, 2) +
                                el(matrix, 3, 0) * el(matrix, 0, 2) * el(matrix, 1, 1);

            inverse[3*4 + 3] =  el(matrix, 0, 0) * el(matrix, 1, 1) * el(matrix, 2, 2) -
                                el(matrix, 0, 0) * el(matrix, 1, 2) * el(matrix, 2, 1) -
                                el(matrix, 1, 0) * el(matrix, 0, 1) * el(matrix, 2, 2) +
                                el(matrix, 1, 0) * el(matrix, 0, 2) * el(matrix, 2, 1) +
                                el(matrix, 2, 0) * el(matrix, 0, 1) * el(matrix, 1, 2) -
                                el(matrix, 2, 0) * el(matrix, 0, 2) * el(matrix, 1, 1);

            // Calculate determinant using the first column of cofactors
            T det = el(matrix, 0, 0) * inverse[0*4 + 0] +
                    el(matrix, 1, 0) * inverse[0*4 + 1] +
                    el(matrix, 2, 0) * inverse[0*4 + 2] +
                    el(matrix, 3, 0) * inverse[0*4 + 3];

            // Check if matrix is invertible
            if (fabs(det) < 1e-40) {
                *success = 0; // Not invertible
                return;
            }

            // Scale cofactors by 1/det to get the inverse
            T invDet = 1.0 / det;
            for (int i = 0; i < 16; ++i) {
                inverse[i] *= invDet;
            }

            *success = 1; // Success
        }
    }
} // namespace

template <class T>
void invert4x4ColMajor(const T* matrix, T* inverse, int* success)
{
    int* successOnDevice;
    OPM_GPU_SAFE_CALL(cudaMalloc(&successOnDevice, sizeof(int)));

    // Launch the kernel with a single thread in a single block
    cuInvert4x4ColMajor<T><<<1, 1>>>(matrix, inverse, successOnDevice);

    // Copy success flag back to host
    OPM_GPU_SAFE_CALL(cudaMemcpy(success, successOnDevice, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    OPM_GPU_SAFE_CALL(cudaFree(successOnDevice));
}

// Explicit template instantiations
template void invert4x4ColMajor<float>(const float* matrix, float* inverse, int* success);
template void invert4x4ColMajor<double>(const double* matrix, double* inverse, int* success);

} // namespace Opm::gpuistl::detail
