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

#include <opm/simulators/linalg/gpuistl/GpuMatrix.hpp>

#include <opm/simulators/linalg/gpuistl/detail/cublas_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/detail/cublas_wrapper.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_matrix_operations.hpp>

#include <fmt/core.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace Opm::gpuistl
{

template <class T>
GpuMatrix<T>::GpuMatrix(size_type rows, size_type cols)
    : m_rows(rows)
    , m_cols(cols)
    , m_cuBlasHandle(detail::CuBlasHandle::getInstance())
{
    if (rows < 1 || cols < 1) {
        OPM_THROW(std::invalid_argument, fmt::format("Invalid matrix dimensions: {}x{}", rows, cols));
    }

    const size_t numElements = rows * cols;
    OPM_GPU_SAFE_CALL(cudaMalloc(&m_dataOnDevice, sizeof(T) * numElements));
}

template <class T>
GpuMatrix<T>::GpuMatrix(size_type rows, size_type cols, const T* dataOnHost)
    : GpuMatrix(rows, cols)
{
    copyFromHost(dataOnHost, rows * cols);
}

template <class T>
GpuMatrix<T>::GpuMatrix(size_type rows, size_type cols, const std::vector<T>& data)
    : GpuMatrix(rows, cols)
{
    if (data.size() != rows * cols) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Data size mismatch: expected {}x{}={} elements, got {}",
                              rows, cols, rows * cols, data.size()));
    }

    // Convert from row-major to column-major and copy to device
    convertAndCopyFromRowMajor(data);
}

template <class T>
GpuMatrix<T>::GpuMatrix(const GpuMatrix<T>& other)
    : m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_cuBlasHandle(detail::CuBlasHandle::getInstance())
{
    const size_t numElements = size();
    OPM_GPU_SAFE_CALL(cudaMalloc(&m_dataOnDevice, sizeof(T) * numElements));
    OPM_GPU_SAFE_CALL(cudaMemcpy(m_dataOnDevice, other.m_dataOnDevice, sizeof(T) * numElements, cudaMemcpyDeviceToDevice));
}

template <class T>
GpuMatrix<T>& GpuMatrix<T>::operator=(const GpuMatrix<T>& other)
{
    if (this != &other) {
        assertSameSize(other);

        const size_t numElements = size();
        OPM_GPU_SAFE_CALL(cudaMemcpy(m_dataOnDevice, other.m_dataOnDevice, sizeof(T) * numElements, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

template <class T>
GpuMatrix<T>::~GpuMatrix()
{
    if (m_dataOnDevice) {
        OPM_GPU_WARN_IF_ERROR(cudaFree(m_dataOnDevice));
        m_dataOnDevice = nullptr;
    }
}

template <class T>
T* GpuMatrix<T>::data()
{
    return m_dataOnDevice;
}

template <class T>
const T* GpuMatrix<T>::data() const
{
    return m_dataOnDevice;
}

template <class T>
typename GpuMatrix<T>::size_type GpuMatrix<T>::rows() const
{
    return m_rows;
}

template <class T>
typename GpuMatrix<T>::size_type GpuMatrix<T>::cols() const
{
    return m_cols;
}

template <class T>
typename GpuMatrix<T>::size_type GpuMatrix<T>::size() const
{
    return m_rows * m_cols;
}

template <class T>
void GpuMatrix<T>::copyFromHost(const T* hostData, size_type numElements)
{
    if (numElements > size()) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Requesting to copy too many elements. Matrix has {} elements, while {} was requested.",
                              size(), numElements));
    }

    // Convert from row-major to column-major and copy to device
    convertAndCopyFromRowMajor(hostData, numElements);
}

template <class T>
void GpuMatrix<T>::copyFromHost(const std::vector<T>& data)
{
    if (data.size() != size()) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Data size mismatch: expected {} elements, got {}",
                              size(), data.size()));
    }

    // Convert from row-major to column-major and copy to device
    convertAndCopyFromRowMajor(data);
}

template <class T>
void GpuMatrix<T>::convertAndCopyFromRowMajor(const std::vector<T>& data)
{
    // Use the raw pointer version
    convertAndCopyFromRowMajor(data.data(), data.size());
}

template <class T>
void GpuMatrix<T>::convertAndCopyFromRowMajor(const T* hostData, size_type numElements)
{
    // Create a temporary vector to convert from row-major to column-major
    std::vector<T> colMajorData(size());

    // Convert row-major to column-major
    for (auto i = 0*m_rows; i < m_rows; ++i) {
        for (auto j = 0*m_cols; j < m_cols; ++j) {
            // hostData[i * m_cols + j] is row-major access
            // colMajorData[j * m_rows + i] is column-major access
            colMajorData[j * m_rows + i] = hostData[i * m_cols + j];
        }
    }

    // Copy to device in column-major order
    OPM_GPU_SAFE_CALL(cudaMemcpy(m_dataOnDevice, colMajorData.data(), numElements * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void GpuMatrix<T>::copyToHost(T* hostData, size_t numElements) const
{
    if (numElements != size()) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Requesting to copy {} elements, but matrix has {} elements",
                              numElements, size()));
    }

    // Convert from column-major to row-major
    convertAndCopyToRowMajor(hostData, numElements);
}

template <class T>
void GpuMatrix<T>::copyToHost(std::vector<T>& data) const
{
    data.resize(size());

    // Convert column-major to row-major
    convertAndCopyToRowMajor(data.data(), data.size());
}

template <class T>
void GpuMatrix<T>::convertAndCopyToRowMajor(std::vector<T>& data) const
{
    // Use the raw pointer version
    convertAndCopyToRowMajor(data.data(), data.size());
}

template <class T>
void GpuMatrix<T>::convertAndCopyToRowMajor(T* hostData, size_type numElements) const
{
    if (numElements != size()) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Requesting to copy {} elements, but matrix has {} elements",
                              size(), numElements));
    }

    // Get data in column-major order
    std::vector<T> colMajorData(size());
    OPM_GPU_SAFE_CALL(cudaMemcpy(colMajorData.data(), m_dataOnDevice, size() * sizeof(T), cudaMemcpyDeviceToHost));

    // Convert column-major to row-major
    for (auto i = 0*m_rows; i < m_rows; ++i) {
        for (auto j = 0*m_cols; j < m_cols; ++j) {
            // colMajorData[j * m_rows + i] is column-major access
            // hostData[i * m_cols + j] is row-major access
            hostData[i * m_cols + j] = colMajorData[j * m_rows + i];
        }
    }
}

template <class T>
std::vector<T> GpuMatrix<T>::asStdVector() const
{
    std::vector<T> result(size());
    convertAndCopyToRowMajor(result.data(), result.size());
    return result;
}

template <class T>
void GpuMatrix<T>::mv(const GpuVector<T>& x, GpuVector<T>& y) const
{
    assertCompatibleVector(x, true);
    assertCompatibleVector(y, false);

    // Matrix is stored internally in column-major format
    // For matrix-vector multiplication: y = Ax
    // In cuBLAS with column-major storage: y = Ax
    T alpha = 1.0;
    T beta = 0.0;

    OPM_CUBLAS_SAFE_CALL(detail::cublasGemv(m_cuBlasHandle.get(),
                                           CUBLAS_OP_N,      // No transpose needed
                                           m_rows,           // Number of rows
                                           m_cols,           // Number of columns
                                           &alpha,           // Alpha value
                                           m_dataOnDevice,   // Matrix data
                                           m_rows,           // Leading dimension
                                           x.data(),         // Input vector
                                           1,                // Stride for x
                                           &beta,            // Beta value
                                           y.data(),         // Output vector
                                           1));              // Stride for y
}

template <class T>
void GpuMatrix<T>::umv(const GpuVector<T>& x, GpuVector<T>& y) const
{
    assertCompatibleVector(x, true);
    assertCompatibleVector(y, false);

    // Use GEMV: y = alpha*A*x + beta*y with alpha=1, beta=1
    T alpha = 1.0;
    T beta = 1.0;

    // Matrix is stored internally in column-major format
    OPM_CUBLAS_SAFE_CALL(detail::cublasGemv(m_cuBlasHandle.get(),
                                           CUBLAS_OP_N,      // No transpose needed
                                           m_rows,           // Number of rows
                                           m_cols,           // Number of columns
                                           &alpha,           // Alpha value
                                           m_dataOnDevice,   // Matrix data
                                           m_rows,           // Leading dimension
                                           x.data(),         // Input vector
                                           1,                // Stride for x
                                           &beta,            // Beta value
                                           y.data(),         // Output vector
                                           1));              // Stride for y
}

template <class T>
void GpuMatrix<T>::usmv(T alpha, const GpuVector<T>& x, GpuVector<T>& y) const
{
    assertCompatibleVector(x, true);
    assertCompatibleVector(y, false);

    // Use GEMV: y = alpha*A*x + beta*y with beta=1
    T beta = 1.0;

    // Matrix is stored internally in column-major format
    OPM_CUBLAS_SAFE_CALL(detail::cublasGemv(m_cuBlasHandle.get(),
                                           CUBLAS_OP_N,      // No transpose needed
                                           m_rows,           // Number of rows
                                           m_cols,           // Number of columns
                                           &alpha,           // Alpha value
                                           m_dataOnDevice,   // Matrix data
                                           m_rows,           // Leading dimension
                                           x.data(),         // Input vector
                                           1,                // Stride for x
                                           &beta,            // Beta value
                                           y.data(),         // Output vector
                                           1));              // Stride for y
}

template <class T>
void GpuMatrix<T>::mm(const GpuMatrix<T>& B, GpuMatrix<T>& C) const
{
    assertCanMultiply(B, C);

    const T alpha = 1.0;
    const T beta = 0.0;

    // Use cuBLAS to multiply matrices (column-major for both A and B)
    // C = alpha * A * B + beta * C
    OPM_CUBLAS_SAFE_CALL(detail::cublasGemm(m_cuBlasHandle.get(),
                                           CUBLAS_OP_N,      // No transpose for A
                                           CUBLAS_OP_N,      // No transpose for B
                                           m_rows,           // Rows of A
                                           B.m_cols,         // Cols of B
                                           m_cols,           // Cols of A / Rows of B
                                           &alpha,           // Alpha value
                                           m_dataOnDevice,   // Matrix A
                                           m_rows,           // Leading dimension of A
                                           B.m_dataOnDevice, // Matrix B
                                           B.m_rows,         // Leading dimension of B
                                           &beta,            // Beta value
                                           C.m_dataOnDevice, // Matrix C
                                           C.m_rows));       // Leading dimension of C
}

template <class T>
void GpuMatrix<T>::mmtv(const GpuVector<T>& x, GpuVector<T>& y) const
{
    // Check that x has the same size as the number of rows in the matrix
    if (x.dim() != detail::to_size_t(m_rows)) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Vector x size mismatch for mmtv: expected {} elements (rows of matrix), got {}",
                            m_rows, x.dim()));
    }

    // Check that y has the same size as the number of columns in the matrix
    if (y.dim() != detail::to_size_t(m_cols)) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Vector y size mismatch for mmtv: expected {} elements (columns of matrix), got {}",
                            m_cols, y.dim()));
    }

    // y -= A^T * x is equivalent to y = y - A^T * x = -1.0 * A^T * x + 1.0 * y
    const T alpha = -1.0;
    const T beta = 1.0;

    // Use cuBLAS to compute y = -A^T * x + y (which is equivalent to y -= A^T * x)
    // so we can use a simple gemv operation with transposed matrix (CUBLAS_OP_T)
    OPM_CUBLAS_SAFE_CALL(detail::cublasGemv(m_cuBlasHandle.get(),
                                           CUBLAS_OP_T,      // Transpose operation needed
                                           m_rows,           // Number of rows
                                           m_cols,           // Number of columns
                                           &alpha,           // Alpha value (-1.0)
                                           m_dataOnDevice,   // Matrix data
                                           m_rows,           // Leading dimension
                                           x.data(),         // Input vector
                                           1,                // Stride for x
                                           &beta,            // Beta value (1.0)
                                           y.data(),         // Output vector
                                           1));              // Stride for y
}

template <class T>
void GpuMatrix<T>::setIdentity()
{
    assertSquare();

    // First set all to zero
    const size_t numElements = size();
    OPM_GPU_SAFE_CALL(cudaMemset(m_dataOnDevice, 0, sizeof(T) * numElements));

    // Set diagonal elements to 1
    // We'll create a temporary vector and use it to set the diagonal
    std::vector<T> diag(m_rows, 1.0);

    // In column-major format, diagonal elements are at positions i*rows + i
    for (size_type i = 0; i < m_rows; ++i) {
        OPM_GPU_SAFE_CALL(cudaMemcpy(m_dataOnDevice + i*m_rows + i, &diag[i], sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <class T>
bool GpuMatrix<T>::invert()
{
    // Special case for 4x4 matrices
    if (m_rows == 4 && m_cols == 4) {
        int success = 0;
        auto inverse = GpuMatrix<T>(4, 4);
        detail::invert4x4ColMajor<T>(m_dataOnDevice, inverse.data(), &success);
        if (success == 1) {
            *this = inverse;
        }
        return success == 1;
    }
    else {
        OPM_THROW(std::invalid_argument, fmt::format("We currently only support inverting 4x4 matrices. Matrix is {}x{}", m_rows, m_cols));
    }
}

template <class T>
GpuMatrix<T> GpuMatrix<T>::transpose() const
{
    // TODO: Do this with a cuBLAS operation

    // Create a new matrix with swapped dimensions
    GpuMatrix<T> result(m_cols, m_rows);

    // Get data in column-major format
    std::vector<T> colMajorData(size());
    copyToHost(colMajorData.data(), colMajorData.size());

    // For a matrix stored in column-major order, transposing means
    // we swap rows and columns in our access pattern
    std::vector<T> transposedData(size());
    for (auto j = 0*m_cols; j < m_cols; ++j) {
        for (auto i = 0*m_rows; i < m_rows; ++i) {
            // In column-major: original[j*m_rows + i] -> transposed[i*m_cols + j]
            transposedData[i * m_cols + j] = colMajorData[j * m_rows + i];
        }
    }

    // Copy data to device (already in the correct column-major format for the result)
    result.copyFromHost(transposedData.data(), transposedData.size());
    return result;
}

template <class T>
GpuMatrix<T>& GpuMatrix<T>::operator*=(T scalar)
{
    // Scale all elements by scalar
    T hostScalar = scalar;

    OPM_CUBLAS_SAFE_CALL(detail::cublasScal(m_cuBlasHandle.get(),
                                           detail::to_int(size()),
                                           &hostScalar,
                                           m_dataOnDevice,
                                           1));
    return *this;
}

// Private helper methods
template <class T>
void GpuMatrix<T>::assertSameSize(const GpuMatrix<T>& other) const
{
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Matrix size mismatch: this is {}x{}, other is {}x{}",
                              m_rows, m_cols, other.m_rows, other.m_cols));
    }
}

template <class T>
void GpuMatrix<T>::assertValidSize(size_type rows, size_type cols) const
{
    if (rows < 1 || cols < 1) {
        OPM_THROW(std::invalid_argument, fmt::format("Invalid matrix dimensions: {}x{}", rows, cols));
    }
}

template <class T>
void GpuMatrix<T>::assertSquare() const
{
    if (m_rows != m_cols) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Operation requires square matrix. Matrix is {}x{}", m_rows, m_cols));
    }
}

template <class T>
void GpuMatrix<T>::assertCanMultiply(const GpuMatrix<T>& B, const GpuMatrix<T>& C) const
{
    if (m_cols != B.m_rows) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Incompatible matrix dimensions for multiplication: {}x{} * {}x{}",
                              m_rows, m_cols, B.m_rows, B.m_cols));
    }

    if (m_rows != C.m_rows || B.m_cols != C.m_cols) {
        OPM_THROW(std::invalid_argument,
                  fmt::format("Result matrix has incorrect dimensions: expected {}x{}, got {}x{}",
                              m_rows, B.m_cols, C.m_rows, C.m_cols));
    }
}

template <class T>
void GpuMatrix<T>::assertCompatibleVector(const GpuVector<T>& v, bool inputVector) const
{
    if (inputVector) {
        // Input vector x should have size == cols
        if (v.dim() != detail::to_size_t(m_cols)) {
            OPM_THROW(std::invalid_argument,
                     fmt::format("Input vector has incompatible size: expected {}, got {}",
                                m_cols, v.dim()));
        }
    } else {
        // Output vector y should have size == rows
        if (v.dim() != detail::to_size_t(m_rows)) {
            OPM_THROW(std::invalid_argument,
                     fmt::format("Output vector has incompatible size: expected {}, got {}",
                                m_rows, v.dim()));
        }
    }
}

// Explicit instantiations
template class GpuMatrix<float>;
template class GpuMatrix<double>;

} // namespace Opm::gpuistl
