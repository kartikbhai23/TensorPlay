#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include "DataPtr.h"
#include "Allocator.h"
#include "IntrusivePtr.h"
#include "Macros.h"
#include "Exception.h"

namespace tensorplay {

// Storage is the single owner of a byte block, except while the block is
// handed to a second owner through "copy on write" (see share_bytes).  The
// shared form lives in a refcount handle rather than in the storage fields, so
// an ordinary tensor pays nothing for the possibility: the holder is only
// created when a block actually gains a second owner, and the accessors are a
// predictable null check in front of the same load.
struct P10_API StorageImpl : public IntrusivePtrTarget {
    struct SharedBytes {
        DataPtr data;
        size_t nbytes = 0;
        Allocator* allocator = nullptr;
        bool resizable = false;

        SharedBytes(DataPtr&& ptr, size_t bytes, Allocator* alloc, bool resizable_)
            : data(std::move(ptr)), nbytes(bytes), allocator(alloc),
              resizable(resizable_) {}

        Device device() const { return data.device_; }
    };

    DataPtr data_ptr;
    size_t nbytes;
    Allocator* allocator;
    bool resizable;

    // Set only while several storages observe the same bytes.  Rebound from
    // const accessors, which is how the first accessor takes a private copy.
    mutable std::shared_ptr<SharedBytes> shared;

    StorageImpl(size_t size, Allocator* allocator, bool resizable = true)
        : nbytes(size), allocator(allocator), resizable(resizable) {
        if (allocator) {
            data_ptr = allocator->allocate(nbytes);
        }
    }

    StorageImpl(size_t size, Allocator* allocator, const Device& device, bool resizable = true)
        : nbytes(size), allocator(allocator), resizable(resizable) {
        if (allocator) {
            data_ptr = allocator->allocate(nbytes, device);
        }
    }
    
    StorageImpl(DataPtr&& ptr, size_t size, Allocator* alloc, bool resizable = false)
        : data_ptr(std::move(ptr)), nbytes(size), allocator(alloc), resizable(resizable) {}

    // Storage over bytes another storage already holds jointly.
    StorageImpl(std::shared_ptr<SharedBytes> bytes)
        : nbytes(0), allocator(nullptr), resizable(false), shared(std::move(bytes)) {}
        
    void* data() {
        return shared ? shared->data.get() : data_ptr.get();
    }
    
    const void* data() const {
        return shared ? shared->data.get() : data_ptr.get();
    }

    Device device() const {
        return shared ? shared->device() : data_ptr.device_;
    }

    size_t bytes() const { return shared ? shared->nbytes : nbytes; }

    Allocator* allocator_of() const {
        return shared ? shared->allocator : allocator;
    }

    bool resizable_p() const { return shared ? shared->resizable : resizable; }

    // Byte handle, for backends that need the deleter context.
    const DataPtr& ptr() const { return shared ? shared->data : data_ptr; }

    bool has_shared() const { return shared != nullptr; }

    // True while another storage still observes the same bytes.
    bool is_shared() const { return shared != nullptr && shared.use_count() > 1; }

    // Hand the bytes to a second owner: both storages reference one holder
    // from here on, and the first of them to touch the bytes takes a private
    // copy.  The block itself is not copied here.
    std::shared_ptr<SharedBytes> share_bytes() {
        if (!shared) {
            shared = std::make_shared<SharedBytes>(
                std::move(data_ptr), nbytes, allocator, resizable);
            data_ptr = DataPtr();
            nbytes = 0;
            resizable = false;
        }
        return shared;
    }

    // Give this storage its own bytes; other holders keep the original block.
    // Called from a const data accessor, so the block is copied at most once
    // per sharing and only for the side that reaches it first.
    void detach_bytes() const {
        if (!shared) return;
        StorageImpl* self = const_cast<StorageImpl*>(this);
        if (shared.use_count() == 1) {
            // Last holder: take the block back instead of copying it.
            self->data_ptr = std::move(shared->data);
            self->nbytes = shared->nbytes;
            self->allocator = shared->allocator;
            self->resizable = shared->resizable;
            shared.reset();
            return;
        }
        const size_t bytes = shared->nbytes;
        Allocator* alloc = shared->allocator;
        if (alloc == nullptr || bytes == 0) {
            // Memory the storage does not own cannot be re-owned privately.
            // Sharing stays; callers that need isolation avoid sharing such
            // storages in the first place.
            shared.reset();
            return;
        }
        const Device device = shared->device();
        const bool resizable_ = shared->resizable;
        DataPtr priv = alloc->allocate(bytes, device);
        copyAllocationBytes(priv.get(), priv.device_, shared->data.get(), device, bytes);
        self->data_ptr = std::move(priv);
        self->nbytes = bytes;
        self->allocator = alloc;
        self->resizable = resizable_;
        shared.reset();
    }
    
    void set_nbytes(size_t new_nbytes) {
        if (shared) detach_bytes();
        if (nbytes == new_nbytes) return;
        if (!resizable || !allocator) {
            TP_THROW(RuntimeError, "Storage is not resizable");
        }
        
        DataPtr new_data = allocator->allocate(new_nbytes, data_ptr.device_);
        
        size_t copy_size = std::min(nbytes, new_nbytes);
        if (data_ptr.get() && new_data.get()) {
            copyAllocationBytes(new_data.get(), new_data.device_,
                                data_ptr.get(), data_ptr.device_, copy_size);
        }
        
        data_ptr = std::move(new_data);
        nbytes = new_nbytes;
    }

protected:
    // Weak observers keep the header alive after the last strong reference;
    // hand back the byte buffer at that point instead of at destruction.
    void release_resources() override {
        shared.reset();
        data_ptr = DataPtr();
        nbytes = 0;
    }
};

} // namespace tensorplay