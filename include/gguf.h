#ifndef GGUF_H
#define GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ==================================================================== *
 * Section 1 — Type enums
 *
 * Verified against the official GGUF spec:
 *   https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 * Gaps are intentional and must never be reused.
 * =================================================================== */

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  /* 4, 5 (Q4_2, Q4_3) removed - do not reuse */
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_BF16 = 30,
  /* 31-33 removed - CPU repack-only, never stored in GGUF */
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  /* 36-38 reserved / repack-only */
  GGML_TYPE_MXFP4 = 39,
  GGML_TYPE_COUNT = 40
};

/* Type tag for every metadata value in the KV store. */
enum gguf_metadata_value_type {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7, /* 1 byte; only 0/1 are valid */
  GGUF_TYPE_STRING =
      8, /* length-prefixed UTF-8 on disk, NUL-terminated in memory */
  GGUF_TYPE_ARRAY = 9, /* element type + count + elements; may nest */
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12
};

/* ==================================================================== *
 * Section 2 — In-memory value representation
 * ==================================================================== */

/*
 * A single parsed metadata value.
 *
 * Arrays are recursive: a value of type ARRAY owns a heap array of
 * gguf_value_t. This lets one parsing path handle scalars, strings,
 * arrays-of-numbers, arrays-of-strings and (rare) arrays-of-arrays.
 */
typedef struct gguf_value gguf_value_t;

struct gguf_value {
  enum gguf_metadata_value_type type;
  union {
    uint8_t v_u8;
    int8_t v_i8;
    uint16_t v_u16;
    int16_t v_i16;
    uint32_t v_u32;
    int32_t v_i32;
    uint64_t v_u64;
    int64_t v_i64;
    float v_f32;
    double v_f64;
    uint8_t v_bool; /* 0 or 1 */
    char *v_str;    /* heap-allocated, NUL-terminated */
    struct {
      enum gguf_metadata_value_type elem_type;
      uint64_t len;
      gguf_value_t *items; /* heap array of length `len` */
    } v_arr;
  } v;
};

typedef struct {
  char *key; /* heap-allocated, NUL-terminated */
  gguf_value_t value;
} gguf_kv_t;

typedef struct {
  uint32_t n_dims;
  uint64_t dims[4]; /* GGUF allows at most 4 dimensions */
  enum ggml_type type;
  uint64_t offset; /* relative to the tensor-data section */
  char *name;      /* heap-allocated, NUL-terminated */
} gguf_tensor_info_t;

typedef struct {
  /* raw header fields (magic is checked at load time and discarded) */
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  /* parsed metadata */
  gguf_kv_t *kv; /* array of length metadata_kv_count */

  /* parsed tensor descriptors (not the tensor bytes themselves) */
  gguf_tensor_info_t *tensors; /* array of length tensor_count */

  /* derived / bookkeeping */
  uint32_t alignment;        /* from general.alignment, else 32 */
  size_t tensor_data_offset; /* absolute file offset of tensor data */

  /* mmap bookkeeping so gguf_free() can unmap cleanly */
  void *mmap_base;
  size_t mmap_size;
  int fd;
} gguf_file_t;

/* ==================================================================== *
 * Section 3 — Public API
 * ==================================================================== */

/* Loads and fully parses a .gguf file (header, all metadata, all
 * tensor descriptors). Tensor data itself stays mmap'd and is not
 * touched until something later reads it.
 *
 * Returns 0 on success, -1 on failure (message printed to stderr).
 * On success the caller must eventually call gguf_free(). */
int gguf_load(const char *path, gguf_file_t *out);

/* Releases everything owned by a gguf_file_t, including the mmap. */
void gguf_free(gguf_file_t *f);

/* Looks up a metadata key by exact name. Returns NULL if absent. */
const gguf_value_t *gguf_find(const gguf_file_t *f, const char *key);

/* Human-readable name for a ggml_type, e.g. "Q4_K".
 * Unknown values are written into `scratch` as "UNKNOWN(<n>)".
 * The scratch buffer avoids static-buffer aliasing problems. */
const char *gguf_type_name(enum ggml_type t, char *scratch, size_t scratch_len);

/* Human-readable name for a metadata value type, e.g. "STRING".
 * Same scratch-buffer rationale as gguf_type_name(). */
const char *gguf_value_type_name(enum gguf_metadata_value_type t, char *scratch,
                                 size_t scratch_len);

/* Prints a human-readable summary of the loaded file to stdout.
 * max_tensors_shown caps the tensor listing (-1 = show all). */
void gguf_print_summary(const gguf_file_t *f, int max_tensors_shown);

#endif /* GGUF_H */