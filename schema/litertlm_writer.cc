// This tool is used to create a LiteRT-LM file from a set of input files
// (tokenizer, tflite model, llm parameters), and metadata.
//
// Example usage:
//
// bazel run
// //third_party/odml/litert_lm/schema:litertlm_writer \
//   -- --output_path=/path/to/output.litertlm \
//   /path/to/tokenizer.spiece \
//   /path/to/model.tflite \
//   /path/to/llm_metadata.pbtext \ (or binary proto via .pb or .proto)
//   /path/to/model2.tflite \
//   --section_metadata="tokenizer:key1=value1,key2=value2;\
//     tflite:key3=123,key4=true;llm_metadata:key5=abc;tflite:z=9.8"

#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "schema/litertlm_writer_utils.h"

ABSL_FLAG(std::string, output_path, "",
          "The path for the output LiteRT-LM file.");

// Flag to handle key-value pairs. Example usage:
// --section_metadata="tokenizer:key1=value1,key2=value2;tflite:key3=123,key4=true"
ABSL_FLAG(std::string, section_metadata, "",
          "Metadata for sections in the format "
          "'section_name:key1=value1,key2=value2;...'. "
          "Supported value types: int32, int64, uint32, uint64, bool, float, "
          "string.");

const char* const ANSI_RESET = "\033[0m";
const char* const ANSI_BOLD_GREEN = "\033[1;32m";
const char* const CAKE_EMOJI_UTF8 = "\xF0\x9F\x8E\x82";  // 🎂 UTF-8 literal

namespace {

// --- Utility function to format numbers with commas ---
std::string PrettyPrintBytes(uint64_t bytes) {
  std::string num_str = std::to_string(bytes);
  const std::size_t len = num_str.length();
  if (len <= 3) {
    return num_str;  // No commas needed
  }

  std::string result = "";
  const std::size_t first_segment_len = len % 3;
  if (first_segment_len == 0) {
    first_segment_len = 3;
  }

  result += num_str.substr(0, first_segment_len);

  for (std::size_t i = first_segment_len; i < len; i += 3) {
    result += ',';
    result += num_str.substr(i, 3);
  }
  return result;
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  std::string section_metadata_str = absl::GetFlag(FLAGS_section_metadata);

  ABSL_LOG(INFO) << "output_path is " << output_path << "\n";
  ABSL_LOG(INFO) << "section_metadata is " << section_metadata_str << "\n";

  std::vector<std::string> command_args;
  // argv[0] is the program name. The first actual argument is argv[1].
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    // Skip any flags that might be present.
    // Flags typically start with "--".
    if (arg.rfind("--", 0) == 0) {
      continue;  // Skip all arguments that start with "--"
    }
    command_args.push_back(arg);
  }
  ABSL_LOG(INFO) << "Collected command_args: ";
  for (const auto& ca : command_args) {
    ABSL_LOG(INFO) << ca;
  }

  return ::litert::lm::schema::LitertLmWrite(command_args, section_metadata_str,
                                             output_path);
}

}  // namespace

int main(int argc, char** argv) {
  absl::Status status = MainHelper(argc, argv);
  std::string output_path_for_message =
      absl::GetFlag(FLAGS_output_path);  // Get it for the success message

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error: " << status.message();
    std::cerr << ANSI_RESET  // Ensure terminal is reset on error output
              << "Error creating LiteRT-LM file: " << status.message()
              << std::endl;
    return 1;
  }

  // Verify output file was created for final status message.
  uint64_t file_size_bytes = 0;
  bool file_size_known = false;
  if (!output_path_for_message.empty()) {
    std::ifstream file(output_path_for_message,
                       std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      std::streamsize size = file.tellg();
      if (size !=
          static_cast<std::streamsize>(-1)) {  // Check for error from tellg
        file_size_bytes = static_cast<uint64_t>(size);
        file_size_known = true;
      } else {
        ABSL_LOG(WARNING) << "Could not determine file size for "
                          << output_path_for_message
                          << " using ifstream (tellg failed).";
      }
      file.close();
    } else {
      ABSL_LOG(WARNING) << "Could not open file to get size: "
                        << output_path_for_message;
    }
  }

  std::string size_info_str = "";
  if (file_size_known) {
    size_info_str = std::string(" and is of size ") + ANSI_BOLD_GREEN +
                    PrettyPrintBytes(file_size_bytes) + ANSI_RESET + " bytes";
  }

  // Success message.
  std::cout << ANSI_RESET  // Reset any previous colors first
            << CAKE_EMOJI_UTF8 << " " << ANSI_BOLD_GREEN << "LiteRT-LM"
            << ANSI_RESET << " file successfully created! Output is at "
            << output_path_for_message << size_info_str << std::endl;

  return 0;
}
