#include "TensorImpl.h"
#include "Allocator.h"
#include "Exception.h"
#include "Storage.h"
#include "Tensor.h"
#include "tensorplay/TensorRedispatchGenerated.h"

namespace tensorplay {

// Resolve the deferred state carried by this tensor's bytes: take a private
// block while another storage still observes it, then apply a pending sign to
// those private bytes.  Logical contents are identical before and after, which
// is what makes both steps safe to run on first access.
//
// Runs at most once per deferral, and a tensor that is never reached pays
// nothing: the last holder simply takes the block back instead of copying.
void TensorImpl::cow_materialize() const {
    if (!storage_.defined()) return;
    StorageImpl* impl = storage_.unsafeGetStorageImpl().get();
    if (impl == nullptr) return;
    impl->detach_bytes();
    if (!impl->take_negation()) return;
    // The bytes now hold the values as stored, and this tensor's logical values
    // are their negation; fold the sign into the bytes so that every later
    // reader (kernels, exports, views) sees the final values directly.  The
    // negation runs on a metadata copy that shares this tensor's storage, and
    // through the out entry point so that no mutation is reported to the
    // version counter: the caller never asked for an in-place update.
    Tensor mirror(make_intrusive<TensorImpl>(*const_cast<TensorImpl*>(this)));
    detail::redispatch_neg_out_function(mirror, mirror);
}

const char* toString(MemoryFormat format) {
    switch (format) {
        case MemoryFormat::Contiguous: return "Contiguous";
        case MemoryFormat::Preserve: return "Preserve";
        case MemoryFormat::ChannelsLast: return "ChannelsLast";
        case MemoryFormat::ChannelsLast3d: return "ChannelsLast3d";
    }
    return "Unknown";
}

TensorImpl::TensorImpl()
    : storage_offset_(0), dtype_(DType::Float32), device_(DeviceType::CPU),
      is_contiguous_(true) {
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device)
    : TensorImpl(sizes, dtype, device, true) {}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype,
                       const Device& device, bool allocate_storage)
    : storage_offset_(0), sizes_and_strides_(sizes), dtype_(dtype), device_(device),
      is_contiguous_(true) {
    // Every dense allocation derives its byte count from these values, so a
    // negative entry would wrap the size_t product into a bogus (huge)
    // allocation or an out-of-range memset rather than failing loudly.
    for (const int64_t s : sizes) {
        if (s < 0) {
            TP_THROW(RuntimeError,
                     "Trying to create tensor with negative dimension ", s);
        }
    }
    int64_t num_elements = sizes_and_strides_.numel();
    (void)num_elements;
    if (allocate_storage) {
        size_t total_bytes = static_cast<size_t>(
            subByteStorageSize(sizes, dtype));
        Allocator* allocator = getAllocator(device.type());
        storage_ = Storage(total_bytes, allocator, device);
        device_ = storage_.device();
    }
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes, strides), dtype_(dtype), device_(device),
      is_contiguous_(false) {
    is_contiguous_ = sizes_and_strides_.is_contiguous();

    for (const int64_t s : sizes) {
        if (s < 0) {
            TP_THROW(RuntimeError,
                     "Trying to create tensor with negative dimension ", s);
        }
    }
    int64_t num_elements = sizes_and_strides_.numel();
    (void)num_elements;
    size_t total_bytes = static_cast<size_t>(
        subByteStorageSize(sizes, dtype));
    Allocator* allocator = getAllocator(device.type());
    storage_ = Storage(total_bytes, allocator, device);
    device_ = storage_.device();
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes), dtype_(dtype),
      device_(storage.device()), is_contiguous_(true) {
    storage_ = std::move(storage);
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset)
    : storage_offset_(storage_offset), sizes_and_strides_(sizes, strides), dtype_(dtype),
      device_(storage.device()), is_contiguous_(false) {
    storage_ = std::move(storage);
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(intrusive_ptr<TensorImpl> transform_value,
                       const std::vector<int64_t>& sizes,
                       const std::vector<int64_t>& strides)
    : storage_offset_(0), sizes_and_strides_(sizes, strides),
      dtype_(transform_value ? transform_value->dtype() : DType::Undefined),
      device_(transform_value ? transform_value->device() : Device(DeviceType::CPU)),
      is_contiguous_(sizes_and_strides_.is_contiguous()),
      transform_value_(std::move(transform_value)) {
    shared_state_ = std::make_shared<SharedState>();
}

// Copy does not carry autograd metadata: copies start fresh, matching
TensorImpl::TensorImpl(const TensorImpl& other)
    : storage_offset_(other.storage_offset_),
      sizes_and_strides_(other.sizes_and_strides_),
      dtype_(other.dtype_),
      device_(other.device_),
      version_counter_(other.version_counter_),
      inference_tensor_(other.inference_tensor_),
      is_contiguous_(other.is_contiguous_),
      memory_format_(other.memory_format_),
      storage_(other.storage_),
      shared_state_(other.shared_state_),
      sparse_state_(other.sparse_state_),
      nested_state_(other.nested_state_),
      transform_value_(other.transform_value_),
      transform_batch_dim_(other.transform_batch_dim_),
      transform_level_(other.transform_level_),
      quantizer_(other.quantizer_) {}
TensorImpl::TensorImpl(TensorImpl&& other) noexcept = default;
TensorImpl& TensorImpl::operator=(const TensorImpl& other) = default;
TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept = default;

void TensorImpl::set_requires_grad(bool requires_grad) {
    if (requires_grad && inference_tensor_ && !InferenceMode::is_enabled()) {
        throw std::runtime_error(
            "Setting requires_grad=True on an inference tensor outside inference mode is not allowed.");
    }
    if (autograd_meta_) {
        autograd_meta_->set_requires_grad(requires_grad);
    }
}

void TensorImpl::copy_metadata_from(const TensorImpl& other) {
    storage_ = other.storage_;
    shared_state_ = other.shared_state_;
    storage_offset_ = other.storage_offset_;
    sizes_and_strides_ = other.sizes_and_strides_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    is_contiguous_ = other.is_contiguous_;
    memory_format_ = other.memory_format_;
    sparse_state_ = other.sparse_state_;
    nested_state_ = other.nested_state_;
    transform_value_ = other.transform_value_;
    transform_batch_dim_ = other.transform_batch_dim_;
    transform_level_ = other.transform_level_;
    quantizer_ = other.quantizer_;
}

} // namespace tensorplay
