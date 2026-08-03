// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"  // NOLINT
#include "zconf.h"  // from @zlib
#include "zlib.h"  // from @zlib

namespace litert {
namespace lm {
namespace schema {

// Abstract base class for section streams
//
// A "section stream" represents a source of data that can be read sequentially.
// This data can be a file, a serialized protocol buffer, or any other
// contiguous block of bytes. The purpose of this abstraction is to allow
// different data sources to be handled uniformly by a reader or writer.
//
// Example Usage:
//
//   // Create a concrete stream (e.g., from a file)
//   std::unique_ptr<SectionStreamBase> stream =
//       std::make_unique<FileBackedSectionStream>("my_data.bin");
//
//   // Prepare the stream
//   absl::Status status = stream->Prepare();
//   if (!status.ok()) {
//     // Handle error
//   }
//
//   // Get the input stream
//   std::istream& istream = stream->GetStream();
//
//   // Read data from the stream
//   std::string data;
//   std::getline(istream, data);
//
//   // Finalize the stream
//   status = stream->Finalize();
//   if (!status.ok()) {
//     // Handle error
//   }
//
// This pattern allows a parser to work with different data sources without
// needing to know the specifics of how each source is handled.
class SectionStreamBase {
 public:
  // Virtual destructor
  virtual ~SectionStreamBase() = default;

  // Prepare: Pure virtual function to prepare the stream.
  virtual absl::Status Prepare() = 0;

  // GetStream: Pure virtual function to get the input stream.
  virtual std::istream& GetStream() = 0;

  // IsReady:  Check if the stream is ready for reading.
  virtual bool IsReady() const = 0;

  // Finalize:  Pure virtual function to finalize the stream.
  virtual absl::Status Finalize() = 0;

  // BufferSize:  Pure virtual function get the size of the streamed buffer.
  virtual size_t BufferSize() const = 0;
};

// A basic derived class for a file-backed stream. Reads the provided file
// during the Prepare(), holds it in an internal buffer, and then provides
// an input string stream for the caller to stream its contents.
class FileBackedSectionStream : public SectionStreamBase {
 public:
  // Constructor: Takes the file path.
  explicit FileBackedSectionStream(const std::string& file_path)
      : file_path_(file_path), buffer_(nullptr), buffer_size_(0) {}

  ~FileBackedSectionStream() override = default;

  // Prepare: Reads the file and prepares the internal buffer.  This function
  // *must* be called before using the stream.
  absl::Status Prepare() override {
    if (buffer_) {
      ABSL_LOG(INFO) << "Buffer already prepared for file: " << file_path_;
      return absl::OkStatus();
    }

    // Clear the stringstream before use.
    stream_.str(std::string());
    stream_.clear();

    std::ifstream file(file_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return absl::InternalError(
          absl::StrCat("Failed to open file: ", file_path_));
    }

    buffer_size_ = static_cast<size_t>(file.tellg());  // Use size_t
    file.seekg(0, std::ios::beg);
    ABSL_DLOG(INFO) << "File size: " << buffer_size_ << " bytes.";

    buffer_.reset(new unsigned char[buffer_size_]);
    if (!file.read(reinterpret_cast<char*>(buffer_.get()), buffer_size_)) {
      buffer_.reset();  // Clean up on error. Set to null
      return absl::InternalError(
          absl::StrCat("Failed to read all data from file: ", file_path_));
    }
    ABSL_DLOG(INFO) << "Successfully read " << buffer_size_
                    << " bytes from file.";

    file.close();
    is_ready_ = true;
    stream_.write(reinterpret_cast<char*>(buffer_.get()), buffer_size_);
    stream_.seekg(0);
    ABSL_DLOG(INFO) << "Internal stringstream prepared.";
    return absl::OkStatus();
  }

  // Expose a stream-like object.
  std::istream& GetStream() override {
    if (!is_ready_) {
      ABSL_LOG(ERROR) << "Attempting to get stream before preparation.";
    }
    return stream_;
  }

  bool IsReady() const override { return is_ready_; }

  size_t BufferSize() const override { return buffer_size_; }

  absl::Status Finalize() override {
    if (buffer_) {
      buffer_.reset();  // Release the memory
      buffer_size_ = 0;
      is_ready_ = false;
      stream_.str(std::string());  // Clear the stringstream
      stream_.clear();             // Clear any error flags
      ABSL_LOG(INFO) << "Buffer finalized and stream reset for file: "
                     << file_path_;
    } else {
      ABSL_LOG(INFO) << "Nothing to finalize. Either Prepare() was not called "
                     << "or Finalize() has already been called.";
    }
    return absl::OkStatus();
  }

 private:
  std::string file_path_;
  std::unique_ptr<unsigned char[]> buffer_;
  size_t buffer_size_;
  bool is_ready_ = false;  // Track preparation state
  std::stringstream stream_;
};

// Class template for a stream backed by a protocol buffer.
// This class is particularly useful when a section of data is directly
// represented as a protocol buffer object in memory. Instead of writing this
// object to a file and then reading it back, this class allows to serialize
// the protocol buffer directly into a stream, which can then be used by the
// writer. This approach is more efficient as it avoids the overhead of file
// I/O and the need for temporary files.
template <typename T>
class ProtoBufSectionStream : public SectionStreamBase {
 public:
  // Constructor: Own a copy of the protocol buffer object, so that
  // this object can guarantee the ability to stream the contents
  // and release the memory upon destruction.
  explicit ProtoBufSectionStream(T proto)
      : proto_(std::move(proto)), is_ready_(false) {}

  ~ProtoBufSectionStream() override = default;

  // Prepare: Serializes the protocol buffer to a string.
  absl::Status Prepare() override {
    if (is_ready_) {
      ABSL_LOG(INFO) << "Stream already prepared for proto.";
      return absl::OkStatus();
    }

    // Clear the stringstream before use.
    stream_.str(std::string());
    stream_.clear();

    // Write directly into the stringstream's buffer.
    if (!proto_.SerializeToOstream(&stream_)) {
      return absl::InternalError("Failed to serialize protocol buffer.");
    }
    serialized_size_ =
        stream_.str()
            .size();  // Get the size from the stringstream's underlying string.
    is_ready_ = true;
    ABSL_LOG(INFO)
        << "Protocol buffer serialized directly to stringstream, size: "
        << serialized_size_ << " bytes.";
    return absl::OkStatus();
  }

  // GetStream: Returns a reference to the internal string stream.
  std::istream& GetStream() override {
    if (!is_ready_) {
      ABSL_LOG(ERROR) << "Attempting to get stream before preparation.";
    }
    return stream_;
  }

  bool IsReady() const override { return is_ready_; }

  absl::Status Finalize() override {
    stream_.str(std::string());
    stream_.clear();
    serialized_size_ = 0;
    is_ready_ = false;
    ABSL_LOG(INFO) << "Stream finalized.";
    return absl::OkStatus();
  }

  size_t BufferSize() const override { return serialized_size_; }

 private:
  T proto_;
  std::stringstream stream_;
  bool is_ready_;
  size_t serialized_size_;
};

class ZlibBackendedSectionStream : public SectionStreamBase {
 public:
  explicit ZlibBackendedSectionStream(
      std::unique_ptr<SectionStreamBase> base_stream)
      : base_stream_(std::move(base_stream)) {}

  absl::Status Prepare() override {
    if (is_ready_) {
      ABSL_LOG(INFO) << "Stream already prepared.";
      return absl::OkStatus();
    }

    RETURN_IF_ERROR(base_stream_->Prepare());  // NOLINT

    // Initialize zlib stream structure
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    if (deflateInit(&strm, Z_DEFAULT_COMPRESSION) != Z_OK) {
      return absl::InternalError("Failed to initialize zlib compression.");
    }

    // Initialize both the compressed and uncompressed data vectors.
    std::vector<char> uncompressed_data(
        std::istreambuf_iterator<char>(base_stream_->GetStream()),
        std::istreambuf_iterator<char>());
    std::vector<char> compressed_data;
    compressed_data.resize(deflateBound(&strm, uncompressed_data.size()));

    strm.next_in = reinterpret_cast<Bytef*>(uncompressed_data.data());
    strm.avail_in = uncompressed_data.size();

    // Compress the data in chunks of 16KB.
    const size_t kZlibChunkSize = 16 * 1024;
    int ret;
    uint64_t compressed_size = 0;
    do {
      strm.next_out =
          reinterpret_cast<Bytef*>(compressed_data.data() + compressed_size);
      strm.avail_out = kZlibChunkSize;

      ret = deflate(&strm, strm.avail_in == 0 ? Z_FINISH : Z_NO_FLUSH);
      if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
        deflateEnd(&strm);
        return absl::InternalError("Compression failed with error code: " +
                                   std::to_string(ret));
      }

      compressed_size += (kZlibChunkSize - strm.avail_out);
    } while (ret != Z_STREAM_END);
    deflateEnd(&strm);

    // Write the uncompressed size
    uint64_t uncompressed_size = uncompressed_data.size();
    zlib_stream_.write(reinterpret_cast<const char*>(&uncompressed_size),
                       sizeof(uncompressed_size));
    zlib_serialized_size_ += sizeof(uncompressed_size);

    // Write the compressed data
    zlib_stream_.write(compressed_data.data(), compressed_size);
    zlib_serialized_size_ += compressed_size;

    is_ready_ = true;
    return absl::OkStatus();
  }

  std::istream& GetStream() override { return zlib_stream_; }

  bool IsReady() const override { return !is_ready_; }

  absl::Status Finalize() override {
    zlib_stream_.str(std::string());
    zlib_stream_.clear();
    zlib_serialized_size_ = 0;
    is_ready_ = false;
    ABSL_LOG(INFO) << "Zlib section stream finalized.";
    return absl::OkStatus();
  }

  size_t BufferSize() const override {
    if (!is_ready_) {
      ABSL_LOG(ERROR) << "Attempting to get stream before preparation.";
    }
    return zlib_serialized_size_;
  }

 private:
  std::unique_ptr<SectionStreamBase> base_stream_;
  std::stringstream zlib_stream_;
  size_t zlib_serialized_size_ = 0;
  bool is_ready_ = false;
};

// String-backed section stream class. This class is useful when a section of
// data is represented as a pure string in memory. Instead of writing this
// object to a file and then reading it back, this class allows to serialize
// the string directly into a stream, which can then be used by the
// writer.
class StringBackedSectionStream : public SectionStreamBase {
 public:
  // Constructor takes a std::string by value to allow for efficient moving.
  explicit StringBackedSectionStream(std::string data)
      : data_(std::move(data)), buffer_size_(0), is_ready_(false) {}

  ~StringBackedSectionStream() override = default;

  // Prepare: Copies the stored string into the internal stringstream.
  absl::Status Prepare() override {
    if (is_ready_) {
      return absl::OkStatus();
    }

    // Set the internal stream's content to our stored string data.
    stream_.str(data_);
    buffer_size_ = data_.size();

    // The data has been copied into the stream's internal buffer,
    // so we can release the memory from our copy.
    data_.clear();
    data_.shrink_to_fit();

    is_ready_ = true;
    return absl::OkStatus();
  }

  // GetStream: Returns a reference to the internal stringstream.
  std::istream& GetStream() override { return stream_; }

  bool IsReady() const override { return is_ready_; }

  size_t BufferSize() const override { return buffer_size_; }

  absl::Status Finalize() override {
    // Clear the stringstream's buffer and reset its state.
    stream_.str(std::string());
    stream_.clear();
    buffer_size_ = 0;
    is_ready_ = false;
    return absl::OkStatus();
  }

 private:
  // Temporarily holds the data until Prepare() is called.
  std::string data_;
  // The stream that will be provided to the consumer.
  std::stringstream stream_;
  size_t buffer_size_;
  bool is_ready_;
};

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_SECTION_H
