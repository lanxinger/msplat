#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

// CFRelease is needed in both C++ and ObjC++ to release retained Metal buffers.
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

enum class DType : uint8_t {
    Float32,
    Int32,
    Int64,
    UInt8,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::Float64: return 8;
    }
}

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
// Ownership: GPU-allocated tensors own their Metal buffer (released on destruction).
// Copies are non-owning aliases. Moves transfer ownership. Views are non-owning.
class MTensor {
public:
    MTensor() = default;

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only) — creates an owning tensor.
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype), _owning(true) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        _buffer = (__bridge void*)buf;  // non-ARC: newBuffer already +1 retained, CFRelease handles release
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (__bridge id<MTLBuffer>)_buffer; }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    // Copy constructor: non-owning alias (shares buffer without retaining).
    MTensor(const MTensor& o)
        : _buffer(o._buffer), _data(o._data), _cpu_data(o._cpu_data),
          _shape(o._shape), _dtype(o._dtype), _numel(o._numel), _owning(false) {}

    // Move constructor: transfers ownership.
    MTensor(MTensor&& o) noexcept
        : _buffer(o._buffer), _data(o._data), _cpu_data(std::move(o._cpu_data)),
          _shape(std::move(o._shape)), _dtype(o._dtype), _numel(o._numel), _owning(o._owning) {
        o._buffer = nullptr;
        o._data = nullptr;
        o._numel = 0;
        o._owning = false;
    }

    // Copy assignment: non-owning alias.
    MTensor& operator=(const MTensor& o) {
        if (this != &o) {
            releaseBuffer();
            _buffer = o._buffer;
            _data = o._data;
            _cpu_data = o._cpu_data;
            _shape = o._shape;
            _dtype = o._dtype;
            _numel = o._numel;
            _owning = false;
        }
        return *this;
    }

    // Move assignment: transfers ownership, releases old buffer if owning.
    MTensor& operator=(MTensor&& o) noexcept {
        if (this != &o) {
            releaseBuffer();
            _buffer = o._buffer;
            _data = o._data;
            _cpu_data = std::move(o._cpu_data);
            _shape = std::move(o._shape);
            _dtype = o._dtype;
            _numel = o._numel;
            _owning = o._owning;
            o._buffer = nullptr;
            o._data = nullptr;
            o._numel = 0;
            o._owning = false;
        }
        return *this;
    }

    ~MTensor() { releaseBuffer(); }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        releaseBuffer();
        _buffer = nullptr;
        _data = nullptr;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // Non-owning — shares the underlying MTLBuffer without retaining it.
    // The caller MUST ensure the parent MTensor outlives all views.
    MTensor view(int64_t n) const {
        MTensor v;
        v._buffer = _buffer;  // shares the buffer (non-owning, _owning stays false)
        v._data = _data;
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        return v;
    }

private:
    void releaseBuffer() {
#ifdef __APPLE__
        if (_owning && _buffer) {
            CFRelease(_buffer);
        }
#endif
        _owning = false;
    }

    void* _buffer = nullptr;  // retained id<MTLBuffer> as void*
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
    bool _owning = false;     // true if this instance owns (must CFRelease) _buffer
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
