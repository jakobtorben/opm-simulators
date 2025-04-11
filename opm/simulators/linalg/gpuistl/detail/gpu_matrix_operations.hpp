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
#ifndef OPM_GPUISTL_GPU_MATRIX_OPERATIONS_HPP
#define OPM_GPUISTL_GPU_MATRIX_OPERATIONS_HPP


namespace Opm::gpuistl::detail
{
/**
 * @brief Inverts a 4x4 matrix on the GPU
 *
 * @param matrix The source 4x4 matrix in column-major format (stored on GPU)
 * @param inverse The destination buffer for the inverted matrix (on GPU)
 * @param success Output parameter to indicate success (will be set to 1) or failure (will be set to 0)
 *
 * @note Both matrices must be stored in column-major format on the GPU
 */
template <class T>
void invert4x4ColMajor(const T* matrix, T* inverse, int* success);

} // namespace Opm::gpuistl::detail

#endif // OPM_GPUISTL_GPU_MATRIX_OPERATIONS_HPP
