#ifndef GGML_TYPES_H
#define GGML_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * GGML tensor type identifiers stored in GGUF tensor descriptors.
 *
 * Values are wire-format ABI: never renumber a value or reuse a removed one.
 * This list matches the current upstream GGML enum.  The GGUF specification
 * documents the persistent subset; GGML may add storage types before that
 * document is updated, so loader validation uses this complete current list.
 */
enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  /* 4, 5: Q4_2 and Q4_3 were removed. */
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
  /* 31-33: removed CPU repack-only types. */
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  /* 36-38: removed CPU repack-only types. */
  GGML_TYPE_MXFP4 = 39,
  GGML_TYPE_NVFP4 = 40,
  GGML_TYPE_Q1_0 = 41,
  GGML_TYPE_Q2_0 = 42,
  GGML_TYPE_COUNT = 43
};

typedef struct {
  const char *name;
  uint32_t block_size; /* logical elements represented by one storage block */
  uint32_t type_size;  /* bytes in one storage block */
} ggml_type_traits_t;

/*
 * Obtains the storage layout for an active GGML type.  Returns false for a
 * removed, unknown, or non-storage type.  This module describes bytes only;
 * it intentionally contains no quantization or dequantization behavior.
 */
bool ggml_get_type_traits(enum ggml_type type, ggml_type_traits_t *out);

/* Returns the canonical name for an active type, or NULL for an invalid one. */
const char *ggml_type_name(enum ggml_type type);

/*
 * Calculates contiguous GGML storage for a tensor shape without overflow.
 * Quantized tensors require dimensions[0] to be a whole number of blocks.
 * `dimensions` may be NULL only when n_dims is zero (a scalar shape).
 * Returns false for invalid types, malformed shapes, or unrepresentable sizes.
 */
bool ggml_tensor_storage_size(enum ggml_type type, uint32_t n_dims,
                              const uint64_t *dimensions, size_t *out_size);

#endif /* GGML_TYPES_H */
