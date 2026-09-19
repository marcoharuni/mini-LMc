#include <stdint.h>
#include <stdio.h>

#include "ggml_types.h"

typedef struct {
  enum ggml_type type;
  uint32_t block_size;
  uint32_t type_size;
} expected_traits_t;

static int failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,     \
              #condition);                                                   \
      failures++;                                                             \
    }                                                                        \
  } while (0)

static void test_all_active_type_layouts(void) {
  static const expected_traits_t expected[] = {
      {GGML_TYPE_F32, 1, 4},       {GGML_TYPE_F16, 1, 2},
      {GGML_TYPE_Q4_0, 32, 18},    {GGML_TYPE_Q4_1, 32, 20},
      {GGML_TYPE_Q5_0, 32, 22},    {GGML_TYPE_Q5_1, 32, 24},
      {GGML_TYPE_Q8_0, 32, 34},    {GGML_TYPE_Q8_1, 32, 36},
      {GGML_TYPE_Q2_K, 256, 84},   {GGML_TYPE_Q3_K, 256, 110},
      {GGML_TYPE_Q4_K, 256, 144},  {GGML_TYPE_Q5_K, 256, 176},
      {GGML_TYPE_Q6_K, 256, 210},  {GGML_TYPE_Q8_K, 256, 292},
      {GGML_TYPE_IQ2_XXS, 256, 66}, {GGML_TYPE_IQ2_XS, 256, 74},
      {GGML_TYPE_IQ3_XXS, 256, 98}, {GGML_TYPE_IQ1_S, 256, 50},
      {GGML_TYPE_IQ4_NL, 32, 18},  {GGML_TYPE_IQ3_S, 256, 110},
      {GGML_TYPE_IQ2_S, 256, 82},  {GGML_TYPE_IQ4_XS, 256, 136},
      {GGML_TYPE_I8, 1, 1},        {GGML_TYPE_I16, 1, 2},
      {GGML_TYPE_I32, 1, 4},       {GGML_TYPE_I64, 1, 8},
      {GGML_TYPE_F64, 1, 8},       {GGML_TYPE_IQ1_M, 256, 56},
      {GGML_TYPE_BF16, 1, 2},      {GGML_TYPE_TQ1_0, 256, 54},
      {GGML_TYPE_TQ2_0, 256, 66},  {GGML_TYPE_MXFP4, 32, 17},
      {GGML_TYPE_NVFP4, 64, 36},   {GGML_TYPE_Q1_0, 128, 18},
      {GGML_TYPE_Q2_0, 64, 18},
  };

  for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
    ggml_type_traits_t traits;
    uint64_t dimensions[2] = {expected[i].block_size * 2, 3};
    size_t size = 0;

    CHECK(ggml_get_type_traits(expected[i].type, &traits));
    CHECK(traits.block_size == expected[i].block_size);
    CHECK(traits.type_size == expected[i].type_size);
    CHECK(ggml_type_name(expected[i].type) != NULL);
    CHECK(ggml_tensor_storage_size(expected[i].type, 2, dimensions, &size));
    CHECK(size == (size_t)6 * expected[i].type_size);
  }
}

static void test_invalid_layouts(void) {
  uint64_t q4_shape[1] = {31};
  uint64_t f32_shape[2] = {1, UINT64_MAX};
  size_t size = 0;

  CHECK(!ggml_get_type_traits((enum ggml_type)4, NULL));
  CHECK(!ggml_get_type_traits((enum ggml_type)33, NULL));
  CHECK(!ggml_get_type_traits((enum ggml_type)GGML_TYPE_COUNT, NULL));
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_Q4_0, 1, q4_shape, &size));
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_F32, 2, f32_shape, &size));
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_F32, 5, q4_shape, &size));
  CHECK(!ggml_tensor_storage_size(GGML_TYPE_F32, 1, NULL, &size));
}

int main(void) {
  test_all_active_type_layouts();
  test_invalid_layouts();
  return failures != 0;
}
