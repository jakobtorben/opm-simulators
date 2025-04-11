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

#ifndef OPM_GPUMATRIX_HEADER_HPP
#define OPM_GPUMATRIX_HEADER_HPP


#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/detail/CuBlasHandle.hpp>
#include <opm/simulators/linalg/gpuistl/detail/safe_conversion.hpp>

#include <cstddef>
#include <fmt/core.h>

#include <vector>

namespace Opm::gpuistl
{

/**
 * @brief The GpuMatrix class is a simple dense matrix class for the GPU.
 *
 * @note This class uses column-major storage internally to be compatible with cuBLAS,
 *       but presents a standard C++ interface. All data passed to or from the GPU
 *       is automatically converted between row-major and column-major formats.
 * @note We currently only support simple raw primitives for T (double and float)
 * @note Integer matrices are not supported for arithmetic operations
 *
 * Example usage:
 *
 * @code{.cpp}
 * // Create a 3x5 matrix
 * GpuMatrix<double> matrix(3, 5);
 *
 * // Fill with data from a vector
 * std::vector<double> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
 * matrix.copyFromHost(data);
 *
 * // Multiply a vector
 * GpuVector<double> vec(5);
 * GpuVector<double> result(3);
 * matrix.mv(vec, result);
 * @endcode
 *
 * @tparam T the type to store. Can be either float or double.
 */
template <typename T>
class GpuMatrix
{
public:
    using field_type = T;
    using size_type = std::size_t;

    /**
     * @brief GpuMatrix creates a dense matrix with the specified dimensions
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    GpuMatrix(size_type rows, size_type cols);

    /**
     * @brief GpuMatrix creates a matrix from raw data
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dataOnHost Pointer to host data
     */
    GpuMatrix(size_type rows, size_type cols, const T* dataOnHost);

    /**
     * @brief GpuMatrix creates a matrix from std::vector
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @param data Vector containing matrix data
     *
     * @note The size of data must be rows * cols
     */
    GpuMatrix(size_type rows, size_type cols, const std::vector<T>& data);

    /**
     * @brief Copy constructor
     *
     * @param other Matrix to copy from
     */
    GpuMatrix(const GpuMatrix<T>& other);

    /**
     * @brief Assignment operator
     *
     * @param other Matrix to copy from
     */
    GpuMatrix& operator=(const GpuMatrix<T>& other);

    /**
     * @brief Destructor
     */
    virtual ~GpuMatrix();

    /**
     * @brief Access raw data pointer
     * @return Pointer to GPU memory
     */
    T* data();

    /**
     * @brief Access raw data pointer (const version)
     * @return Const pointer to GPU memory
     */
    const T* data() const;

    /**
     * @brief Get the number of rows
     * @return Number of rows
     */
    size_type rows() const;

    /**
     * @brief Get the number of columns
     * @return Number of columns
     */
    size_type cols() const;

    /**
     * @brief Get the total number of elements in the matrix
     * @return Number of elements (rows * cols)
     */
    size_type size() const;

    /**
     * @brief Copy data from host to device
     *
     * @param hostData Pointer to host data in row-major format
     * @param numElements Number of elements to copy
     *
     * @note Input data is expected to be in row-major format and will be
     *       converted to column-major format internally for the GPU
     */
    void copyFromHost(const T* hostData, size_type numElements);

    /**
     * @brief Copy data from host to device
     *
     * @param data Vector containing matrix data in row-major format
     *
     * @note The size of data must match rows * cols
     * @note Input data is expected to be in row-major format and will be
     *       converted to column-major format internally for the GPU
     */
    void copyFromHost(const std::vector<T>& data);

    /**
     * @brief Copy data from a matrix
     *
     * @param matrix Matrix to copy from
     *
     * @note The dimensions must match this matrix
     * @note Matrix is automatically interpreted as row-major format
     *       and will be converted to column-major format internally for the GPU
     * @tparam MatrixType any matrix type with accessor operator[][] for elements
     */
    template <typename MatrixType>
    void copyFromHost(const MatrixType& matrix)
    {
        if (detail::to_size_t(matrix.N()) != m_rows || detail::to_size_t(matrix.M()) != m_cols) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Matrix size mismatch. Matrix is {}x{}, but source matrix is {}x{}",
                                  m_rows, m_cols, matrix.N(), matrix.M()));
        }

        std::vector<T> data(m_rows * m_cols);
        for (auto i = 0*m_rows; i < m_rows; ++i) {
            for (auto j = 0*m_cols; j < m_cols; ++j) {
                data[i * m_cols + j] = matrix[i][j];
            }
        }
        copyFromHost(data);
    }

    /**
     * @brief Copy data from device to host
     *
     * @param hostData Pointer to host memory
     * @param numElements Number of elements to copy
     *
     * @note Output data will be converted from column-major (GPUMatrix internal format)
     *       to row-major format for consistent behavior with other methods.
     */
    void copyToHost(T* hostData, size_t numElements) const;

    /**
     * @brief Copy data from device to host
     *
     * @param data Vector to store matrix data
     *
     * @note Output data will be converted from column-major (GPU's internal format)
     *       to row-major format
     */
    void copyToHost(std::vector<T>& data) const;

    /**
     * @brief Copy data from device to a matrix
     *
     * @param matrix Matrix to store the data
     *
     * @note The dimensions must match this matrix
     * @note Output data will be converted from column-major (GPUMatrix internal format)
     *       to row-major format as required by the destination matrix
     * @tparam MatrixType any matrix type with accessor operator[][] for elements
     */
    template <typename MatrixType>
    void copyToHost(MatrixType& matrix) const
    {
        if (detail::to_size_t(matrix.N()) != m_rows || detail::to_size_t(matrix.M()) != m_cols) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Matrix size mismatch. Matrix is {}x{}, but destination matrix is {}x{}",
                                  m_rows, m_cols, matrix.N(), matrix.M()));
        }

        std::vector<T> data(m_rows * m_cols);
        copyToHost(data);

        for (auto i = 0*m_rows; i < m_rows; ++i) {
            for (auto j = 0*m_cols; j < m_cols; ++j) {
                matrix[i][j] = data[i * m_cols + j];
            }
        }
    }

    /**
     * @brief Get a copy of matrix data as a std::vector
     *
     * @return Copy of matrix data in row-major format
     */
    std::vector<T> asStdVector() const;

    /**
     * @brief Matrix-vector multiplication: y = A*x
     *
     * @param x Input vector
     * @param y Output vector
     */
    void mv(const GpuVector<T>& x, GpuVector<T>& y) const;

    /**
     * @brief Matrix-vector update multiplication: y = A*x + y
     *
     * @param x Input vector
     * @param y Input/output vector
     */
    void umv(const GpuVector<T>& x, GpuVector<T>& y) const;

    /**
     * @brief Matrix-vector scaled update multiplication: y = alpha*A*x + y
     *
     * @param alpha Scaling factor
     * @param x Input vector
     * @param y Input/output vector
     */
    void usmv(T alpha, const GpuVector<T>& x, GpuVector<T>& y) const;

    /**
     * @brief Matrix-matrix multiplication: C = A*B
     *
     * @param B Right-hand matrix
     * @param C Result matrix
     */
    void mm(const GpuMatrix<T>& B, GpuMatrix<T>& C) const;

    /**
     * @brief Matrix-matrix transpose vector operation: y -= A^T * x
     *
     * This performs y -= A^T * x where A is this matrix and A^T is its transpose.
     * This is equivalent to subtracting from y the result of multiplying the
     * transpose of this matrix by the vector x.
     *
     * @param x Input vector
     * @param y Output vector that will be updated
     */
    void mmtv(const GpuVector<T>& x, GpuVector<T>& y) const;

    /**
     * @brief Set matrix to identity
     */
    void setIdentity();

    /**
     * @brief Invert the matrix
     *
     * @return true if successful, false if matrix is singular
     */
    bool invert();

    /**
     * @brief Create the transpose of this matrix
     *
     * @return Transposed matrix
     */
    GpuMatrix<T> transpose() const;

    /**
     * @brief Scale matrix by a scalar: A = alpha*A
     *
     * @param scalar Scaling factor
     * @return Reference to this matrix
     */
    GpuMatrix<T>& operator*=(T scalar);

private:
    T* m_dataOnDevice = nullptr;
    const size_type m_rows;
    const size_type m_cols;
    detail::CuBlasHandle& m_cuBlasHandle;

    /**
     * @brief Convert from row-major to column-major and copy to device
     *
     * @param data Matrix data in row-major order
     *
     * @note This is a helper method that handles the conversion from row-major
     *       (standard C++ format) to column-major (CUDA/cuBLAS format) before
     *       copying to the device. All public methods use this internally.
     */
    void convertAndCopyFromRowMajor(const std::vector<T>& data);

    /**
     * @brief Convert from row-major to column-major and copy directly to device
     *
     * @param hostData Pointer to host data in row-major order
     * @param numElements Number of elements to copy
     *
     * @note This is a helper method that handles the conversion from row-major
     *       (standard C++ format) to column-major (CUDA/cuBLAS format) before
     *       copying to the device. All public methods use this internally.
     */
    void convertAndCopyFromRowMajor(const T* hostData, size_type numElements);

    /**
     * @brief Convert from column-major storage to row-major format
     *
     * @param data Vector to store matrix data in row-major order
     *
     * @note This is a helper method that handles the conversion from column-major
     *       (CUDA/cuBLAS format) to row-major (standard C++ format) when copying
     *       data from the device to host.
     */
    void convertAndCopyToRowMajor(std::vector<T>& data) const;

    /**
     * @brief Convert from column-major storage to row-major format
     *
     * @param hostData Pointer to host memory to store matrix data in row-major order
     * @param numElements Number of elements to copy
     *
     * @note This is a helper method that handles the conversion from column-major
     *       (CUDA/cuBLAS format) to row-major (standard C++ format) when copying
     *       data from the device to host.
     */
    void convertAndCopyToRowMajor(T* hostData, size_type numElements) const;

    void assertSameSize(const GpuMatrix<T>& other) const;
    void assertValidSize(size_type rows, size_type cols) const;
    void assertSquare() const;
    void assertCanMultiply(const GpuMatrix<T>& B, const GpuMatrix<T>& C) const;
    void assertCompatibleVector(const GpuVector<T>& v, bool inputVector) const;
};

} // namespace Opm::gpuistl

#endif // OPM_GPUMATRIX_HEADER_HPP
