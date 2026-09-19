#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gguf.h"

/* ============================================================================
 * Bounds-safe binary reading
 * ============================================================================
 */

/* Read a little-endian uint32, bounds-checked.
 * Returns 0 and sets *out on success.
 * Returns -1 on bounds violation or other error.
 * On success, updates cursor and remaining. */
static int read_u32_safe(const uint8_t **cursor, size_t *remaining,
                         uint32_t *out) {
  if (!cursor || !remaining || !out || *remaining < 4) {
    return -1;
  }
  uint32_t v;
  memcpy(&v, *cursor, 4);
  *cursor += 4;
  *remaining -= 4;
  *out = v;
  return 0;
}

/* Read a little-endian uint64, bounds-checked.
 * Returns 0 and sets *out on success.
 * Returns -1 on bounds violation or other error.
 * On success, updates cursor and remaining. */
static int read_u64_safe(const uint8_t **cursor, size_t *remaining,
                         uint64_t *out) {
  if (!cursor || !remaining || !out || *remaining < 8) {
    return -1;
  }
  uint64_t v;
  memcpy(&v, *cursor, 8);
  *cursor += 8;
  *remaining -= 8;
  *out = v;
  return 0;
}

/* Read a uint8, bounds-checked. */
static int read_u8_safe(const uint8_t **cursor, size_t *remaining,
                        uint8_t *out) {
  if (!cursor || !remaining || !out || *remaining < 1) {
    return -1;
  }
  *out = **cursor;
  *cursor += 1;
  *remaining -= 1;
  return 0;
}

/* Read a GGUF string (8-byte length prefix + UTF-8 bytes).
 * Returns a heap-allocated, NUL-terminated string on success.
 * Returns NULL on failure (bounds violation, allocation failure, overflow).
 * On success, updates cursor and remaining. */
static char *read_gguf_string_safe(const uint8_t **cursor, size_t *remaining) {
  if (!cursor || !remaining || *remaining < 8) {
    return NULL;
  }

  uint64_t len;
  memcpy(&len, *cursor, 8);
  *cursor += 8;
  *remaining -= 8;

  /* Prevent allocation overflow: len + 1 for NUL terminator */
  if (len > SIZE_MAX - 1 || len > *remaining) {
    return NULL;
  }

  char *s = malloc(len + 1);
  if (!s) {
    return NULL;
  }

  memcpy(s, *cursor, len);
  s[len] = '\0';
  *cursor += len;
  *remaining -= len;

  return s;
}

/* Overflow-safe alignment calculation.
 * Returns the smallest k >= offset such that k % alignment == 0.
 * Returns SIZE_MAX on overflow. */
static size_t align_up_safe(size_t offset, size_t alignment) {
  if (alignment == 0) {
    return SIZE_MAX;
  }

  size_t remainder = offset % alignment;
  if (remainder == 0) {
    return offset;
  }

  size_t padding = alignment - remainder;

  /* Check for overflow: offset + padding */
  if (padding > SIZE_MAX - offset) {
    return SIZE_MAX;
  }

  return offset + padding;
}

/* ============================================================================
 * Metadata value parsing and freeing
 * ============================================================================
 */

/* Forward declare free_value for recursive array parsing */
static void free_value(gguf_value_t *v);

/* Recursively parse one metadata value, with bounds checking.
 * Returns 0 on success, -1 on failure.
 * The caller owns the resulting gguf_value_t and must free it with
 * free_value().
 * On failure, *out is left in an indeterminate state and must still be freed.
 */
static int parse_value(const uint8_t **cursor, size_t *remaining,
                       enum gguf_metadata_value_type type, gguf_value_t *out) {
  if (!cursor || !remaining || !out) {
    return -1;
  }

  memset(out, 0, sizeof(*out));
  out->type = type;

  switch (type) {
  case GGUF_TYPE_UINT8: {
    uint8_t v;
    if (read_u8_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_u8 = v;
    break;
  }

  case GGUF_TYPE_INT8: {
    uint8_t v;
    if (read_u8_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_i8 = (int8_t)v;
    break;
  }

  case GGUF_TYPE_UINT16: {
    if (*remaining < 2) {
      return -1;
    }
    uint16_t v;
    memcpy(&v, *cursor, 2);
    out->v.v_u16 = v;
    *cursor += 2;
    *remaining -= 2;
    break;
  }

  case GGUF_TYPE_INT16: {
    if (*remaining < 2) {
      return -1;
    }
    int16_t v;
    memcpy(&v, *cursor, 2);
    out->v.v_i16 = v;
    *cursor += 2;
    *remaining -= 2;
    break;
  }

  case GGUF_TYPE_UINT32: {
    uint32_t v;
    if (read_u32_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_u32 = v;
    break;
  }

  case GGUF_TYPE_INT32: {
    uint32_t v;
    if (read_u32_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_i32 = (int32_t)v;
    break;
  }

  case GGUF_TYPE_FLOAT32: {
    uint32_t raw;
    if (read_u32_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    memcpy(&out->v.v_f32, &raw, 4);
    break;
  }

  case GGUF_TYPE_BOOL: {
    uint8_t v;
    if (read_u8_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_bool = v ? 1 : 0;
    break;
  }

  case GGUF_TYPE_STRING: {
    char *s = read_gguf_string_safe(cursor, remaining);
    if (!s) {
      return -1;
    }
    out->v.v_str = s;
    break;
  }

  case GGUF_TYPE_ARRAY: {
    uint32_t elem_type_raw;
    uint64_t arr_len;

    /* Read element type (as uint32) */
    if (read_u32_safe(cursor, remaining, &elem_type_raw) != 0) {
      return -1;
    }

    /* Validate element type */
    enum gguf_metadata_value_type elem_type =
        (enum gguf_metadata_value_type)elem_type_raw;
    switch (elem_type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
    case GGUF_TYPE_BOOL:
    case GGUF_TYPE_STRING:
    case GGUF_TYPE_ARRAY:
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
      break;
    default:
      fprintf(stderr, "Invalid array element type: %u\n", elem_type_raw);
      return -1;
    }

    /* Read array length */
    if (read_u64_safe(cursor, remaining, &arr_len) != 0) {
      return -1;
    }

    out->v.v_arr.elem_type = elem_type;
    out->v.v_arr.len = arr_len;
    out->v.v_arr.items = NULL;

    if (arr_len > 0) {
      /* Prevent allocation overflow */
      if (arr_len > SIZE_MAX / sizeof(gguf_value_t)) {
        fprintf(stderr, "Array too large: %lu elements\n", arr_len);
        return -1;
      }

      out->v.v_arr.items = calloc(arr_len, sizeof(gguf_value_t));
      if (!out->v.v_arr.items) {
        perror("calloc");
        return -1;
      }

      /* Parse each element, with full cleanup on error */
      for (uint64_t i = 0; i < arr_len; i++) {
        if (parse_value(cursor, remaining, elem_type, &out->v.v_arr.items[i]) !=
            0) {
          /* Free all successfully parsed items */
          for (uint64_t j = 0; j < i; j++) {
            free_value(&out->v.v_arr.items[j]);
          }
          free(out->v.v_arr.items);
          out->v.v_arr.items = NULL;
          return -1;
        }
      }
    }
    break;
  }

  case GGUF_TYPE_UINT64: {
    uint64_t v;
    if (read_u64_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_u64 = v;
    break;
  }

  case GGUF_TYPE_INT64: {
    uint64_t v;
    if (read_u64_safe(cursor, remaining, &v) != 0) {
      return -1;
    }
    out->v.v_i64 = (int64_t)v;
    break;
  }

  case GGUF_TYPE_FLOAT64: {
    uint64_t raw;
    if (read_u64_safe(cursor, remaining, &raw) != 0) {
      return -1;
    }
    memcpy(&out->v.v_f64, &raw, 8);
    break;
  }

  default:
    fprintf(stderr, "Unknown metadata value type: %u\n", (unsigned)type);
    return -1;
  }

  return 0;
}

/* Free a parsed value recursively. Safe to call with NULL.
 * After this, the value is zeroed. */
static void free_value(gguf_value_t *v) {
  if (!v) {
    return;
  }
  switch (v->type) {
  case GGUF_TYPE_STRING:
    if (v->v.v_str) {
      free(v->v.v_str);
      v->v.v_str = NULL;
    }
    break;
  case GGUF_TYPE_ARRAY:
    if (v->v.v_arr.items) {
      for (uint64_t i = 0; i < v->v.v_arr.len; i++) {
        free_value(&v->v.v_arr.items[i]);
      }
      free(v->v.v_arr.items);
      v->v.v_arr.items = NULL;
    }
    break;
  default:
    break;
  }
  memset(v, 0, sizeof(*v));
}

/* ============================================================================
 * Main GGUF loader
 * ============================================================================
 */

/* Loads and validates a GGUF v3 file.
 * Returns 0 on success, -1 on failure.
 * On success, out is fully initialized and mmap'd; caller must gguf_free().
 * On failure, out is partially initialized; caller should still gguf_free(). */
int gguf_load(const char *path, gguf_file_t *out) {
  memset(out, 0, sizeof(*out));
  out->fd = -1;

  /* Open file */
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror("open");
    return -1;
  }

  /* Get file size */
  struct stat st;
  if (fstat(fd, &st) < 0) {
    perror("fstat");
    close(fd);
    return -1;
  }

  size_t file_size = (size_t)st.st_size;

  /* Reject empty files */
  if (file_size < 24) {
    fprintf(stderr, "File too small: %zu bytes (minimum 24 for header)\n",
            file_size);
    close(fd);
    return -1;
  }

  /* Memory map the file */
  uint8_t *base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    perror("mmap");
    close(fd);
    return -1;
  }

  out->mmap_base = base;
  out->mmap_size = file_size;
  out->fd = fd;

  /* Parse with bounds checking */
  const uint8_t *cursor = base;
  size_t remaining = file_size;

  /* --- Parse header --- */
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  if (read_u32_safe(&cursor, &remaining, &magic) != 0 ||
      read_u32_safe(&cursor, &remaining, &version) != 0 ||
      read_u64_safe(&cursor, &remaining, &tensor_count) != 0 ||
      read_u64_safe(&cursor, &remaining, &metadata_kv_count) != 0) {
    fprintf(stderr, "Header truncated\n");
    gguf_free(out);
    return -1;
  }

  /* Validate magic */
  if (magic != 0x46554747) { /* "GGUF" */
    fprintf(stderr, "Not a GGUF file (bad magic: 0x%08X)\n", magic);
    gguf_free(out);
    return -1;
  }

  /* Validate version */
  if (version != 3) {
    fprintf(stderr, "Unsupported GGUF version: %u (expected 3)\n", version);
    gguf_free(out);
    return -1;
  }

  out->version = version;
  out->tensor_count = tensor_count;
  out->metadata_kv_count = metadata_kv_count;

  /* --- Parse metadata KV section --- */
  out->kv = calloc(metadata_kv_count, sizeof(gguf_kv_t));
  if (metadata_kv_count > 0 && !out->kv) {
    perror("calloc");
    gguf_free(out);
    return -1;
  }

  for (uint64_t i = 0; i < metadata_kv_count; i++) {
    /* Read key */
    char *key = read_gguf_string_safe(&cursor, &remaining);
    if (!key) {
      fprintf(stderr, "Failed to read metadata key %lu\n", i);
      gguf_free(out);
      return -1;
    }

    /* Read value type */
    uint32_t vtype_raw;
    if (read_u32_safe(&cursor, &remaining, &vtype_raw) != 0) {
      fprintf(stderr, "Failed to read value type for key '%s'\n", key);
      free(key);
      gguf_free(out);
      return -1;
    }

    enum gguf_metadata_value_type vtype =
        (enum gguf_metadata_value_type)vtype_raw;

    out->kv[i].key = key;
    if (parse_value(&cursor, &remaining, vtype, &out->kv[i].value) != 0) {
      fprintf(stderr, "Failed to parse metadata value for key '%s'\n", key);
      gguf_free(out);
      return -1;
    }
  }

  /* --- Parse tensor info section --- */
  out->tensors = calloc(tensor_count, sizeof(gguf_tensor_info_t));
  if (tensor_count > 0 && !out->tensors) {
    perror("calloc");
    gguf_free(out);
    return -1;
  }

  for (uint64_t i = 0; i < tensor_count; i++) {
    /* Read tensor name */
    char *name = read_gguf_string_safe(&cursor, &remaining);
    if (!name) {
      fprintf(stderr, "Failed to read tensor name %lu\n", i);
      gguf_free(out);
      return -1;
    }

    /* Read dimension count */
    uint32_t n_dims;
    if (read_u32_safe(&cursor, &remaining, &n_dims) != 0) {
      fprintf(stderr, "Failed to read n_dims for tensor '%s'\n", name);
      free(name);
      gguf_free(out);
      return -1;
    }

    /* Validate dimension count */
    if (n_dims > 4) {
      fprintf(stderr, "Tensor '%s' has %u dimensions (max 4)\n", name, n_dims);
      free(name);
      gguf_free(out);
      return -1;
    }

    /* Read dimensions */
    for (uint32_t d = 0; d < n_dims; d++) {
      uint64_t dim;
      if (read_u64_safe(&cursor, &remaining, &dim) != 0) {
        fprintf(stderr, "Failed to read dimension %u for tensor '%s'\n", d,
                name);
        free(name);
        gguf_free(out);
        return -1;
      }
      out->tensors[i].dims[d] = dim;
    }

    /* Read tensor type */
    uint32_t ttype_raw;
    if (read_u32_safe(&cursor, &remaining, &ttype_raw) != 0) {
      fprintf(stderr, "Failed to read tensor type for '%s'\n", name);
      free(name);
      gguf_free(out);
      return -1;
    }

    enum ggml_type ttype = (enum ggml_type)ttype_raw;

    /* Validate type */
    ggml_type_traits_t traits;
    if (!ggml_get_type_traits(ttype, &traits)) {
      fprintf(stderr, "Unknown or unsupported tensor type %u for '%s'\n",
              ttype_raw, name);
      free(name);
      gguf_free(out);
      return -1;
    }

    /* Validate tensor storage size and block alignment */
    size_t storage_size;
    if (!ggml_tensor_storage_size(ttype, n_dims, out->tensors[i].dims,
                                  &storage_size)) {
      fprintf(stderr, "Invalid tensor shape for '%s' (type %u)\n", name,
              ttype_raw);
      free(name);
      gguf_free(out);
      return -1;
    }

    /* Read tensor offset */
    uint64_t offset;
    if (read_u64_safe(&cursor, &remaining, &offset) != 0) {
      fprintf(stderr, "Failed to read tensor offset for '%s'\n", name);
      free(name);
      gguf_free(out);
      return -1;
    }

    out->tensors[i].name = name;
    out->tensors[i].n_dims = n_dims;
    out->tensors[i].type = ttype;
    out->tensors[i].offset = offset;
  }

  /* --- Read and validate alignment --- */
  out->alignment = 32; /* Default */
  const gguf_value_t *align_val = gguf_find(out, "general.alignment");
  if (align_val) {
    if (align_val->type != GGUF_TYPE_UINT32) {
      fprintf(stderr, "general.alignment is not UINT32\n");
      gguf_free(out);
      return -1;
    }

    uint32_t align_candidate = align_val->v.v_u32;

    /* Validate alignment: must be power of 2 and >= 1 */
    if (align_candidate == 0 ||
        (align_candidate & (align_candidate - 1)) != 0) {
      fprintf(stderr, "Invalid general.alignment=%u (must be power of 2)\n",
              align_candidate);
      gguf_free(out);
      return -1;
    }

    out->alignment = align_candidate;
  }

  /* --- Calculate tensor data offset --- */
  size_t current_pos = (size_t)(cursor - base);
  size_t aligned_offset = align_up_safe(current_pos, out->alignment);

  if (aligned_offset == SIZE_MAX) {
    fprintf(stderr, "Alignment overflow\n");
    gguf_free(out);
    return -1;
  }

  out->tensor_data_offset = aligned_offset;

  /* --- Validate tensor byte ranges --- */
  for (uint64_t i = 0; i < tensor_count; i++) {
    /* Calculate storage size */
    size_t storage_size;
    if (!ggml_tensor_storage_size(out->tensors[i].type, out->tensors[i].n_dims,
                                  out->tensors[i].dims, &storage_size)) {
      fprintf(stderr, "Invalid storage size for tensor '%s'\n",
              out->tensors[i].name);
      gguf_free(out);
      return -1;
    }

    /* Calculate absolute offset: tensor_data_offset + relative offset */
    uint64_t rel_offset = out->tensors[i].offset;
    if (rel_offset > SIZE_MAX - out->tensor_data_offset) {
      fprintf(stderr, "Tensor offset overflow for '%s'\n",
              out->tensors[i].name);
      gguf_free(out);
      return -1;
    }

    size_t abs_offset = out->tensor_data_offset + rel_offset;

    /* Check storage_size doesn't overflow when added to abs_offset */
    if (storage_size > SIZE_MAX - abs_offset) {
      fprintf(stderr, "Tensor storage size overflow for '%s'\n",
              out->tensors[i].name);
      gguf_free(out);
      return -1;
    }

    size_t end_offset = abs_offset + storage_size;

    /* Verify entire tensor fits within file */
    if (end_offset > file_size) {
      fprintf(stderr,
              "Tensor '%s' extends past EOF: needs bytes [%zu, %zu) in file "
              "of %zu bytes\n",
              out->tensors[i].name, abs_offset, end_offset, file_size);
      gguf_free(out);
      return -1;
    }
  }

  return 0;
}

/* ============================================================================
 * Cleanup and utility functions
 * ============================================================================
 */

/* Release all resources owned by a GGUF file. Safe to call multiple times.
 * After this, the structure is zeroed and fd is set to -1. */
void gguf_free(gguf_file_t *f) {
  if (!f) {
    return;
  }

  if (f->kv) {
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
      if (f->kv[i].key) {
        free(f->kv[i].key);
        f->kv[i].key = NULL;
      }
      free_value(&f->kv[i].value);
    }
    free(f->kv);
    f->kv = NULL;
  }

  if (f->tensors) {
    for (uint64_t i = 0; i < f->tensor_count; i++) {
      if (f->tensors[i].name) {
        free(f->tensors[i].name);
        f->tensors[i].name = NULL;
      }
    }
    free(f->tensors);
    f->tensors = NULL;
  }

  if (f->mmap_base) {
    munmap(f->mmap_base, f->mmap_size);
    f->mmap_base = NULL;
  }

  if (f->fd >= 0) {
    close(f->fd);
    f->fd = -1;
  }

  memset(f, 0, sizeof(*f));
  f->fd = -1;
}

/* Linear search for a metadata key. Returns NULL if not found. */
const gguf_value_t *gguf_find(const gguf_file_t *f, const char *key) {
  if (!f || !f->kv || !key) {
    return NULL;
  }
  for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
    if (f->kv[i].key && strcmp(f->kv[i].key, key) == 0) {
      return &f->kv[i].value;
    }
  }
  return NULL;
}

/* Human-readable name for a GGML type.
 * For unknown values, writes "UNKNOWN(<n>)" into scratch buffer.
 * scratch_len must be >= 32. */
const char *gguf_type_name(enum ggml_type t, char *scratch,
                           size_t scratch_len) {
  switch (t) {
  case GGML_TYPE_F32:
    return "F32";
  case GGML_TYPE_F16:
    return "F16";
  case GGML_TYPE_Q4_0:
    return "Q4_0";
  case GGML_TYPE_Q4_1:
    return "Q4_1";
  case GGML_TYPE_Q5_0:
    return "Q5_0";
  case GGML_TYPE_Q5_1:
    return "Q5_1";
  case GGML_TYPE_Q8_0:
    return "Q8_0";
  case GGML_TYPE_Q8_1:
    return "Q8_1";
  case GGML_TYPE_Q2_K:
    return "Q2_K";
  case GGML_TYPE_Q3_K:
    return "Q3_K";
  case GGML_TYPE_Q4_K:
    return "Q4_K";
  case GGML_TYPE_Q5_K:
    return "Q5_K";
  case GGML_TYPE_Q6_K:
    return "Q6_K";
  case GGML_TYPE_Q8_K:
    return "Q8_K";
  case GGML_TYPE_IQ2_XXS:
    return "IQ2_XXS";
  case GGML_TYPE_IQ2_XS:
    return "IQ2_XS";
  case GGML_TYPE_IQ3_XXS:
    return "IQ3_XXS";
  case GGML_TYPE_IQ1_S:
    return "IQ1_S";
  case GGML_TYPE_IQ4_NL:
    return "IQ4_NL";
  case GGML_TYPE_IQ3_S:
    return "IQ3_S";
  case GGML_TYPE_IQ2_S:
    return "IQ2_S";
  case GGML_TYPE_IQ4_XS:
    return "IQ4_XS";
  case GGML_TYPE_I8:
    return "I8";
  case GGML_TYPE_I16:
    return "I16";
  case GGML_TYPE_I32:
    return "I32";
  case GGML_TYPE_I64:
    return "I64";
  case GGML_TYPE_F64:
    return "F64";
  case GGML_TYPE_IQ1_M:
    return "IQ1_M";
  case GGML_TYPE_BF16:
    return "BF16";
  case GGML_TYPE_TQ1_0:
    return "TQ1_0";
  case GGML_TYPE_TQ2_0:
    return "TQ2_0";
  case GGML_TYPE_MXFP4:
    return "MXFP4";
  case GGML_TYPE_NVFP4:
    return "NVFP4";
  case GGML_TYPE_Q1_0:
    return "Q1_0";
  case GGML_TYPE_Q2_0:
    return "Q2_0";
  default:
    if (scratch && scratch_len > 0) {
      snprintf(scratch, scratch_len, "UNKNOWN(%d)", (int)t);
    }
    return scratch ? scratch : "UNKNOWN";
  }
}

/* Human-readable name for a metadata value type.
 * For unknown values, writes "UNKNOWN(<n>)" into scratch buffer.
 * scratch_len must be >= 32. */
const char *gguf_value_type_name(enum gguf_metadata_value_type t, char *scratch,
                                 size_t scratch_len) {
  switch (t) {
  case GGUF_TYPE_UINT8:
    return "UINT8";
  case GGUF_TYPE_INT8:
    return "INT8";
  case GGUF_TYPE_UINT16:
    return "UINT16";
  case GGUF_TYPE_INT16:
    return "INT16";
  case GGUF_TYPE_UINT32:
    return "UINT32";
  case GGUF_TYPE_INT32:
    return "INT32";
  case GGUF_TYPE_FLOAT32:
    return "FLOAT32";
  case GGUF_TYPE_BOOL:
    return "BOOL";
  case GGUF_TYPE_STRING:
    return "STRING";
  case GGUF_TYPE_ARRAY:
    return "ARRAY";
  case GGUF_TYPE_UINT64:
    return "UINT64";
  case GGUF_TYPE_INT64:
    return "INT64";
  case GGUF_TYPE_FLOAT64:
    return "FLOAT64";
  default:
    if (scratch && scratch_len > 0) {
      snprintf(scratch, scratch_len, "UNKNOWN(%u)", (unsigned)t);
    }
    return scratch ? scratch : "UNKNOWN";
  }
}

/* Print a single metadata value recursively (helper for gguf_print_summary) */
static void print_value(const gguf_value_t *v, int indent) {
  if (!v) {
    return;
  }

  char scratch[64];

  switch (v->type) {
  case GGUF_TYPE_UINT8:
    printf("%u", v->v.v_u8);
    break;
  case GGUF_TYPE_INT8:
    printf("%d", v->v.v_i8);
    break;
  case GGUF_TYPE_UINT16:
    printf("%u", v->v.v_u16);
    break;
  case GGUF_TYPE_INT16:
    printf("%d", v->v.v_i16);
    break;
  case GGUF_TYPE_UINT32:
    printf("%u", v->v.v_u32);
    break;
  case GGUF_TYPE_INT32:
    printf("%d", v->v.v_i32);
    break;
  case GGUF_TYPE_FLOAT32:
    printf("%g", v->v.v_f32);
    break;
  case GGUF_TYPE_UINT64:
    printf("%lu", v->v.v_u64);
    break;
  case GGUF_TYPE_INT64:
    printf("%ld", v->v.v_i64);
    break;
  case GGUF_TYPE_FLOAT64:
    printf("%g", v->v.v_f64);
    break;
  case GGUF_TYPE_BOOL:
    printf("%s", v->v.v_bool ? "true" : "false");
    break;
  case GGUF_TYPE_STRING:
    printf("\"%s\"", v->v.v_str ? v->v.v_str : "(null)");
    break;
  case GGUF_TYPE_ARRAY: {
    const uint64_t shown = v->v.v_arr.len < 8 ? v->v.v_arr.len : 8;

    printf("[");
    for (uint64_t i = 0; i < shown; i++) {
      if (i > 0) {
        printf(", ");
      }
      print_value(&v->v.v_arr.items[i], indent + 1);
    }
    if (v->v.v_arr.len > shown) {
      printf(", ... (%lu more)", v->v.v_arr.len - shown);
    }
    printf("]");
    break;
  }
  default:
    printf("<%s>", gguf_value_type_name(v->type, scratch, sizeof(scratch)));
    break;
  }
}

/* Print a summary of a loaded GGUF file.
 * max_tensors_shown limits the tensor listing (-1 = show all). */
void gguf_print_summary(const gguf_file_t *f, int max_tensors_shown) {
  if (!f) {
    return;
  }

  printf("GGUF version %u\n", f->version);
  printf("Tensor count:  %lu\n", f->tensor_count);
  printf("Metadata KV:   %lu\n", f->metadata_kv_count);
  printf("Alignment:     %u\n", f->alignment);
  printf("Tensor data at: 0x%zx\n", f->tensor_data_offset);

  printf("\n--- Metadata ---\n");
  for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
    printf("%s: ", f->kv[i].key);
    print_value(&f->kv[i].value, 0);
    printf("\n");
  }

  printf("\n--- Tensors ---\n");
  uint64_t shown =
      (max_tensors_shown < 0) ? f->tensor_count : (uint64_t)max_tensors_shown;
  if (shown > f->tensor_count) {
    shown = f->tensor_count;
  }

  for (uint64_t i = 0; i < shown; i++) {
    const gguf_tensor_info_t *t = &f->tensors[i];
    char scratch[32];
    const char *tname = gguf_type_name(t->type, scratch, sizeof(scratch));

    printf("%s: type=%s, dims=[", t->name, tname);
    for (uint32_t d = 0; d < t->n_dims; d++) {
      printf("%lu", t->dims[d]);
      if (d < t->n_dims - 1) {
        printf(", ");
      }
    }
    printf("], offset=%lu\n", t->offset);
  }

  if (shown < f->tensor_count) {
    printf("... (%lu more tensors)\n", f->tensor_count - shown);
  }
}
