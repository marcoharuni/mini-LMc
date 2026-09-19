#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gguf.h"

/* Test harness */
static int test_count = 0;
static int test_passed = 0;
static int test_failed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    test_count++;                                                              \
    printf("\nTest %d: %s\n", test_count, name);                               \
  } while (0)

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (condition) {                                                           \
      printf("  ✓ %s\n", message);                                             \
      test_passed++;                                                           \
    } else {                                                                   \
      printf("  ✗ %s\n", message);                                             \
      test_failed++;                                                           \
    }                                                                          \
  } while (0)

/* Helper: write a file to disk and return path in temporary buffer */
static const char *write_test_file(const uint8_t *data, size_t size) {
  static char path[256];
  static int counter = 0;
  snprintf(path, sizeof(path), "/tmp/gguf_test_%d.gguf", counter++);

  int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    perror("open");
    return NULL;
  }

  if (write(fd, data, size) != (ssize_t)size) {
    perror("write");
    close(fd);
    return NULL;
  }

  close(fd);
  return path;
}

/* Helper: clean up test file */
static void cleanup_test_file(const char *path) {
  if (path) {
    unlink(path);
  }
}

/* Helper: construct a minimal valid GGUF header */
static void write_header(uint8_t *buf, uint32_t version, uint64_t tensor_count,
                         uint64_t metadata_count) {
  uint32_t magic = 0x46554747; /* "GGUF" in little-endian */
  memcpy(buf + 0, &magic, 4);
  memcpy(buf + 4, &version, 4);
  memcpy(buf + 8, &tensor_count, 8);
  memcpy(buf + 16, &metadata_count, 8);
}

/* Test: valid minimal GGUF with no metadata or tensors */
static void test_minimal_valid_gguf(void) {
  TEST("Minimal valid GGUF (no metadata, no tensors)");

  uint8_t buf[24];
  write_header(buf, 3, 0, 0);

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result == 0, "gguf_load succeeded");
  CHECK(f.version == 3, "Version is 3");
  CHECK(f.tensor_count == 0, "Tensor count is 0");
  CHECK(f.metadata_kv_count == 0, "Metadata count is 0");
  CHECK(f.alignment == 32, "Default alignment is 32");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: file too small (truncated header) */
static void test_truncated_header(void) {
  TEST("Truncated header (< 24 bytes)");

  uint8_t buf[16];
  memset(buf, 0, sizeof(buf));

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (truncated)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: empty file */
static void test_empty_file(void) {
  TEST("Empty file");

  const char *path = write_test_file(NULL, 0);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (empty)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: bad magic number */
static void test_bad_magic(void) {
  TEST("Bad magic number");

  uint8_t buf[24];
  uint32_t bad_magic = 0x12345678;
  uint32_t version = 3;
  uint64_t zeros = 0;

  memcpy(buf + 0, &bad_magic, 4);
  memcpy(buf + 4, &version, 4);
  memcpy(buf + 8, &zeros, 8);
  memcpy(buf + 16, &zeros, 8);

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (bad magic)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: unsupported GGUF version */
static void test_unsupported_version(void) {
  TEST("Unsupported GGUF version (v2)");

  uint8_t buf[24];
  write_header(buf, 2, 0, 0);

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (bad version)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: truncated metadata key */
static void test_truncated_metadata_key(void) {
  TEST("Truncated metadata key (incomplete length prefix)");

  uint8_t buf[24 + 4];        /* header + partial length */
  write_header(buf, 3, 0, 1); /* Declare 1 metadata entry */
  buf[24] = 0xFF;             /* Partial length prefix (need 8 bytes) */

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (truncated metadata)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: invalid metadata type tag */
static void test_invalid_metadata_type(void) {
  TEST("Invalid metadata type tag (unknown enum value)");

  uint8_t buf[24 + 8 + 4 + 4 + 4]; /* header + key len + key + vtype + invalid
                                      vtype */
  write_header(buf, 3, 0, 1);

  /* Write a valid key "test" (length=4) */
  uint64_t key_len = 4;
  memcpy(buf + 24, &key_len, 8);
  memcpy(buf + 32, "test", 4);

  /* Write an invalid value type (999) */
  uint32_t bad_type = 999;
  memcpy(buf + 36, &bad_type, 4);

  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (invalid type)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: truncated tensor descriptor */
static void test_truncated_tensor_descriptor(void) {
  TEST("Truncated tensor descriptor (incomplete n_dims)");

  uint8_t buf[37]; /* header(24) + name_len(8) + "model"(5) = 37 bytes total */
  write_header(buf, 3, 1, 0); /* Declare 1 tensor, 0 metadata */

  /* Write tensor name length (5 bytes for "model") */
  uint64_t name_len = 5;
  memcpy(buf + 24, &name_len, 8);
  memcpy(buf + 32, "model", 5);

  /* Only write to 37 bytes, truncate before n_dims (which would start at 37) */
  const char *path = write_test_file(buf, sizeof(buf));
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (truncated tensor)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: too many dimensions */
static void test_too_many_dimensions(void) {
  TEST("Tensor with too many dimensions (> 4)");

  uint8_t buf[256];
  write_header(buf, 3, 1, 0);

  size_t pos = 24;

  /* Tensor name: "model" */
  uint64_t name_len = 5;
  memcpy(buf + pos, &name_len, 8);
  pos += 8;
  memcpy(buf + pos, "model", 5);
  pos += 5;

  /* n_dims: 5 (too many) */
  uint32_t n_dims = 5;
  memcpy(buf + pos, &n_dims, 4);
  pos += 4;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (too many dims)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: invalid tensor type */
static void test_invalid_tensor_type(void) {
  TEST("Tensor with invalid/unknown GGML type");

  uint8_t buf[256];
  write_header(buf, 3, 1, 0);

  size_t pos = 24;

  /* Tensor name: "model" */
  uint64_t name_len = 5;
  memcpy(buf + pos, &name_len, 8);
  pos += 8;
  memcpy(buf + pos, "model", 5);
  pos += 5;

  /* n_dims: 1 */
  uint32_t n_dims = 1;
  memcpy(buf + pos, &n_dims, 4);
  pos += 4;

  /* dimension[0]: 10 */
  uint64_t dim = 10;
  memcpy(buf + pos, &dim, 8);
  pos += 8;

  /* type: 999 (invalid) */
  uint32_t bad_type = 999;
  memcpy(buf + pos, &bad_type, 4);
  pos += 4;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (invalid tensor type)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: valid tensor with F32 type */
static void test_valid_f32_tensor(void) {
  TEST("Valid F32 tensor (10 elements = 40 bytes)");

  uint8_t buf[256];
  write_header(buf, 3, 1, 0);

  size_t pos = 24;

  /* Tensor name: "weights" */
  uint64_t name_len = 7;
  memcpy(buf + pos, &name_len, 8);
  pos += 8;
  memcpy(buf + pos, "weights", 7);
  pos += 7;

  /* n_dims: 1 */
  uint32_t n_dims = 1;
  memcpy(buf + pos, &n_dims, 4);
  pos += 4;

  /* dimension[0]: 10 */
  uint64_t dim = 10;
  memcpy(buf + pos, &dim, 8);
  pos += 8;

  /* type: F32 (0) */
  uint32_t type = 0; /* GGML_TYPE_F32 */
  memcpy(buf + pos, &type, 4);
  pos += 4;

  /* offset: 0 (relative to tensor data start) */
  uint64_t offset = 0;
  memcpy(buf + pos, &offset, 8);
  pos += 8;

  /* Align to 32 and add some tensor data */
  size_t aligned_pos = pos;
  if (aligned_pos % 32 != 0) {
    aligned_pos = pos + (32 - (pos % 32));
  }

  /* Add 40 bytes of tensor data (10 * 4 bytes for F32) */
  size_t total_size = aligned_pos + 40;

  const char *path = write_test_file(buf, total_size);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result == 0, "gguf_load succeeded");
  if (result == 0) {
    CHECK(f.tensor_count == 1, "Tensor count is 1");
    CHECK(f.tensors[0].n_dims == 1, "Tensor has 1 dimension");
    CHECK(f.tensors[0].dims[0] == 10, "Dimension 0 is 10");
    CHECK(f.tensors[0].type == 0, "Type is F32");
  }

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: tensor offset extends past EOF */
static void test_tensor_extends_past_eof(void) {
  TEST("Tensor offset + storage extends past EOF");

  uint8_t buf[256];
  write_header(buf, 3, 1, 0);

  size_t pos = 24;

  /* Tensor name: "weights" */
  uint64_t name_len = 7;
  memcpy(buf + pos, &name_len, 8);
  pos += 8;
  memcpy(buf + pos, "weights", 7);
  pos += 7;

  /* n_dims: 1 */
  uint32_t n_dims = 1;
  memcpy(buf + pos, &n_dims, 4);
  pos += 4;

  /* dimension[0]: 100 (will need 400 bytes for F32) */
  uint64_t dim = 100;
  memcpy(buf + pos, &dim, 8);
  pos += 8;

  /* type: F32 */
  uint32_t type = 0;
  memcpy(buf + pos, &type, 4);
  pos += 4;

  /* offset: 1000 (points way past end of file) */
  uint64_t offset = 1000;
  memcpy(buf + pos, &offset, 8);
  pos += 8;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (tensor past EOF)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: quantized tensor with non-aligned dimension */
static void test_quantized_non_aligned_dim(void) {
  TEST("Q4_0 tensor with non-block-aligned dimension (31)");

  uint8_t buf[256];
  write_header(buf, 3, 1, 0);

  size_t pos = 24;

  /* Tensor name: "model" */
  uint64_t name_len = 5;
  memcpy(buf + pos, &name_len, 8);
  pos += 8;
  memcpy(buf + pos, "model", 5);
  pos += 5;

  /* n_dims: 1 */
  uint32_t n_dims = 1;
  memcpy(buf + pos, &n_dims, 4);
  pos += 4;

  /* dimension[0]: 31 (Q4_0 requires multiple of 32) */
  uint64_t dim = 31;
  memcpy(buf + pos, &dim, 8);
  pos += 8;

  /* type: Q4_0 (2) */
  uint32_t type = 2;
  memcpy(buf + pos, &type, 4);
  pos += 4;

  /* offset: 0 */
  uint64_t offset = 0;
  memcpy(buf + pos, &offset, 8);
  pos += 8;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (non-aligned quantized dim)");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: general.alignment metadata with valid power-of-2 value */
static void test_valid_alignment_metadata(void) {
  TEST("Valid general.alignment = 64 (power of 2)");

  uint8_t buf[256];
  write_header(buf, 3, 0, 1); /* 0 tensors, 1 metadata */

  size_t pos = 24;

  /* Metadata key: "general.alignment" */
  const char *key = "general.alignment";
  uint64_t key_len = strlen(key);
  memcpy(buf + pos, &key_len, 8);
  pos += 8;
  memcpy(buf + pos, key, key_len);
  pos += key_len;

  /* Value type: UINT32 (4) */
  uint32_t vtype = 4;
  memcpy(buf + pos, &vtype, 4);
  pos += 4;

  /* Value: 64 */
  uint32_t alignment = 64;
  memcpy(buf + pos, &alignment, 4);
  pos += 4;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result == 0, "gguf_load succeeded");
  CHECK(f.alignment == 64, "Alignment is 64");

  gguf_free(&f);
  cleanup_test_file(path);
}

/* Test: general.alignment with invalid value (not power of 2) */
static void test_invalid_alignment_metadata(void) {
  TEST("Invalid general.alignment = 63 (not power of 2)");

  uint8_t buf[256];
  write_header(buf, 3, 0, 1);

  size_t pos = 24;

  /* Metadata key: "general.alignment" */
  const char *key = "general.alignment";
  uint64_t key_len = strlen(key);
  memcpy(buf + pos, &key_len, 8);
  pos += 8;
  memcpy(buf + pos, key, key_len);
  pos += key_len;

  /* Value type: UINT32 */
  uint32_t vtype = 4;
  memcpy(buf + pos, &vtype, 4);
  pos += 4;

  /* Value: 63 (not a power of 2) */
  uint32_t alignment = 63;
  memcpy(buf + pos, &alignment, 4);
  pos += 4;

  const char *path = write_test_file(buf, pos);
  if (!path) {
    CHECK(0, "Failed to create test file");
    return;
  }

  gguf_file_t f;
  int result = gguf_load(path, &f);

  CHECK(result != 0, "gguf_load failed (invalid alignment)");

  gguf_free(&f);
  cleanup_test_file(path);
}

int main(void) {
  printf("=== GGUF Loader Test Suite ===\n\n");

  /* Valid file tests */
  test_minimal_valid_gguf();
  test_valid_f32_tensor();
  test_valid_alignment_metadata();

  /* Truncation/size tests */
  test_empty_file();
  test_truncated_header();
  test_truncated_metadata_key();
  test_truncated_tensor_descriptor();

  /* Corruption tests */
  test_bad_magic();
  test_unsupported_version();
  test_invalid_metadata_type();
  test_invalid_tensor_type();

  /* Validation tests */
  test_too_many_dimensions();
  test_quantized_non_aligned_dim();
  test_tensor_extends_past_eof();
  test_invalid_alignment_metadata();

  /* Summary */
  printf("\n=== Test Summary ===\n");
  printf("Total:  %d\n", test_count);
  printf("Passed: %d\n", test_passed);
  printf("Failed: %d\n", test_failed);

  return test_failed != 0 ? 1 : 0;
}
