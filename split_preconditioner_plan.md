
# DILU Preconditioner Split Plan

## 1. Code Refactoring

### 1.1 Create New Header Files
- Move the parallel version to `opm/simulators/linalg/MultithreadDILU.hpp`
- Update `opm/simulators/linalg/DILU.hpp` to only include the serial version
- update CMakeLists_files.cmake to include the new files

### 1.2 Implementation Split
- Extract the serial implementation of DILU from the current `MultithreadDILU` class
- Create `DILU` with only necessary components for serial execution
- Keep `MultithreadDILU` with parallel-specific code and data structures
- Remove runtime thread checking and use explicit class selection instead

### 1.3 Class Structure
- `DILU`: Simpler implementation with minimal data structures
  - Maintain only `Dinv_` vector and reference to matrix
  - Keep only `serialUpdate()` and `serialApply()` methods
- `MultithreadDILU`: Full parallel implementation
  - Keep reordering structures
  - Keep graph coloring logic
  - Maintain level sets for parallel processing
  - Keep parallel update and apply methods


## 2. Test Implementation

### 2.1 Create a simple test in test_dilu_split.cpp
- Create basic test cases for `DILU`
  - Use one simple example that constructs the two preconditioners
  - See example in test_dilu.cpp, but dont make a complex test like this
- Create basic tests for `MultithreadDILU`
  - Verify results match serial version
- Add comparison tests between the two implementations

## 3. Factory Updates

### 3.1 Update PreconditionerFactory_impl.hpp
- Add new creator for "MultithreadDILU" in both sequential and parallel implementations
- Update existing "DILU" creator to use the new `SerialDILU` class
- Update AMG implementations that use DILU as smoother


## 4. Documentation Updates

### 4.1 Update Comments
- Add proper documentation for both classes
- Document when to use which implementation
- Update linear solver documentation

## 5. PR Description

### 5.1 Write PR Description
- Explain motivation for the split
- Document behavioral changes
- Note performance implications
- Include tests and verification approach
- Add any caveats or potential issues

## 6. Implementation Steps

1. Create the new header files with initial implementation
2. Split the implementations between the two files
3. Write test cases
4. Update factory code to use both implementations
5. Update documentation
6. Final review and PR submission
