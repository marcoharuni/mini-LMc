#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gguf.h"

// ---------------------------------------------------------------------------
// Little-endian readers (GGUF is little-endian on disk)
// ---------------------------------------------------------------------------
static uint32_t read_u32(const uint8_t **p) {
  uint32_t v;
  memcpy(&v, *p, 4);
  *p += 4;
  return v;
}

static uint64_t read_u64(const uint8_t **p) {
  uint64_t v;
  memcpy(&v, *p, 8);
  *p += 8;
  return v;
}

// Read a GGUF string (length-prefixed, no NUL), return a NUL-terminated heap
// copy.
static char *read_gguf_string(const uint8_t **p) {
  uint64_t len = read_u64(p);
  char *s = malloc(len + 1);
  if (!s)
    return NULL;
  memcpy(s, *p, len);
  s[len] = '\0';
  *p += len;
  return s;
}

// ---------------------------------------------------------------------------
// Alignment helper
// ---------------------------------------------------------------------------
static uint64_t align_up(uint64_t offset, uint64_t alignment) {
  return offset + (alignment - (offset % alignment)) % alignment;
}

// ---------------------------------------------------------------------------
// Recursively parse one metadata value.
// Returns 0 on success, -1 on failure.
// The caller owns the resulting gguf_value_t and must free it with
// free_value().
// ---------------------------------------------------------------------------
static int parse_value(const uint8_t **cursor,
                       enum gguf_metadata_value_type type, gguf_value_t *out) {
  memset(out, 0, sizeof(*out));
  out->type = type;

  switch (type) {
  case GGUF_TYPE_UINT8:
    out->v.v_u8 = **cursor;
    *cursor += 1;
    break;

  case GGUF_TYPE_INT8:
    out->v.v_i8 = (int8_t) * *cursor;
    *cursor += 1;
    break;

  case GGUF_TYPE_UINT16: {
    uint16_t v;
    memcpy(&v, *cursor, 2);
    out->v.v_u16 = v;
    *cursor += 2;
    break;
  }

  case GGUF_TYPE_INT16: {
    int16_t v;
    memcpy(&v, *cursor, 2);
    out->v.v_i16 = v;
    *cursor += 2;
    break;
  }

  case GGUF_TYPE_UINT32:
    out->v.v_u32 = read_u32(cursor);
    break;

  case GGUF_TYPE_INT32:
    out->v.v_i32 = (int32_t)read_u32(cursor);
    break;

  case GGUF_TYPE_FLOAT32: {
    uint32_t raw = read_u32(cursor);
    memcpy(&out->v.v_f32, &raw, 4);
    break;
  }

  case GGUF_TYPE_BOOL:
    out->v.v_bool = **cursor ? 1 : 0;
    *cursor += 1;
    break;

  case GGUF_TYPE_STRING: {
    char *s = read_gguf_string(cursor);
    if (!s)
      return -1;
    out->v.v_str = s;
    break;
  }

  case GGUF_TYPE_ARRAY: {
    out->v.v_arr.elem_type = (enum gguf_metadata_value_type)read_u32(cursor);
    out->v.v_arr.len = read_u64(cursor);
    out->v.v_arr.items = NULL;

    if (out->v.v_arr.len > 0) {
      out->v.v_arr.items = calloc(out->v.v_arr.len, sizeof(gguf_value_t));
      if (!out->v.v_arr.items)
        return -1;

      for (uint64_t i = 0; i < out->v.v_arr.len; i++) {
        if (parse_value(cursor, out->v.v_arr.elem_type,
                        &out->v.v_arr.items[i]) != 0) {
          // Free already-parsed items (best-effort)
          for (uint64_t j = 0; j < i; j++) {
            // We'll add a free_value helper later; for now just leak
            // on error path — acceptable for a first pass.
          }
          free(out->v.v_arr.items);
          out->v.v_arr.items = NULL;
          return -1;
        }
      }
    }
    break;
  }

  case GGUF_TYPE_UINT64:
    out->v.v_u64 = read_u64(cursor);
    break;

  case GGUF_TYPE_INT64:
    out->v.v_i64 = (int64_t)read_u64(cursor);
    break;

  case GGUF_TYPE_FLOAT64: {
    uint64_t raw = read_u64(cursor);
    memcpy(&out->v.v_f64, &raw, 8);
    break;
  }

  default:
    fprintf(stderr, "Unknown metadata value type: %d\n", type);
    return -1;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Free a parsed value recursively.
// ---------------------------------------------------------------------------
static void free_value(gguf_value_t *v) {
  if (!v)
    return;
  switch (v->type) {
  case GGUF_TYPE_STRING:
    free(v->v.v_str);
    break;
  case GGUF_TYPE_ARRAY:
    if (v->v.v_arr.items) {
      for (uint64_t i = 0; i < v->v.v_arr.len; i++) {
        free_value(&v->v.v_arr.items[i]);
      }
      free(v->v.v_arr.items);
    }
    break;
  default:
    break;
  }
}

// ---------------------------------------------------------------------------
// gguf_load — mmap + parse header + metadata + tensor descriptors
// Returns 0 on success, -1 on failure.
// ---------------------------------------------------------------------------
int gguf_load(const char *path, gguf_file_t *out) {
  memset(out, 0, sizeof(*out));
  out->fd = -1;

  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror("open");
    return -1;
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    perror("fstat");
    close(fd);
    return -1;
  }

  uint8_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    perror("mmap");
    close(fd);
    return -1;
  }

  out->mmap_base = base;
  out->mmap_size = st.st_size;
  out->fd = fd;

  const uint8_t *cursor = base;

  // --- Header ---
  uint32_t magic = read_u32(&cursor);
  uint32_t version = read_u32(&cursor);
  uint64_t tensor_count = read_u64(&cursor);
  uint64_t metadata_kv_count = read_u64(&cursor);

  if (magic != 0x46554747) {
    fprintf(stderr, "Not a GGUF file (bad magic: 0x%08X)\n", magic);
    gguf_free(out);
    return -1;
  }

  if (version != 3) {
    fprintf(stderr, "Unsupported GGUF version: %u (expected 3)\n", version);
    gguf_free(out);
    return -1;
  }

  out->version = version;
  out->tensor_count = tensor_count;
  out->metadata_kv_count = metadata_kv_count;

  // --- Metadata KV section ---
  out->kv = calloc(metadata_kv_count, sizeof(gguf_kv_t));
  if (!out->kv) {
    perror("calloc");
    gguf_free(out);
    return -1;
  }

  for (uint64_t i = 0; i < metadata_kv_count; i++) {
    char *key = read_gguf_string(&cursor);
    if (!key) {
      gguf_free(out);
      return -1;
    }

    enum gguf_metadata_value_type vtype =
        (enum gguf_metadata_value_type)read_u32(&cursor);

    out->kv[i].key = key;
    if (parse_value(&cursor, vtype, &out->kv[i].value) != 0) {
      gguf_free(out);
      return -1;
    }
  }

  // --- Tensor info section ---
  out->tensors = calloc(tensor_count, sizeof(gguf_tensor_info_t));
  if (!out->tensors) {
    perror("calloc");
    gguf_free(out);
    return -1;
  }

  for (uint64_t i = 0; i < tensor_count; i++) {
    char *name = read_gguf_string(&cursor);
    if (!name) {
      gguf_free(out);
      return -1;
    }

    uint32_t n_dims = read_u32(&cursor);
    if (n_dims > 4) {
      fprintf(stderr, "Tensor '%s' has %u dimensions (max 4)\n", name, n_dims);
      free(name);
      gguf_free(out);
      return -1;
    }

    for (uint32_t d = 0; d < n_dims; d++) {
      out->tensors[i].dims[d] = read_u64(&cursor);
    }

    enum ggml_type ttype = (enum ggml_type)read_u32(&cursor);
    uint64_t offset = read_u64(&cursor);

    out->tensors[i].name = name;
    out->tensors[i].n_dims = n_dims;
    out->tensors[i].type = ttype;
    out->tensors[i].offset = offset;
  }

  // --- Alignment (from general.alignment, else 32) ---
  out->alignment = 32;
  const gguf_value_t *align_val = gguf_find(out, "general.alignment");
  if (align_val && align_val->type == GGUF_TYPE_UINT32) {
    out->alignment = align_val->v.v_u32;
    if (out->alignment < 8 || (out->alignment % 8) != 0) {
      fprintf(stderr, "Warning: general.alignment=%u is not a multiple of 8\n",
              out->alignment);
    }
  }

  // --- Tensor data offset: align the cursor position ---
  size_t current_pos = (size_t)(cursor - base);
  out->tensor_data_offset = (size_t)align_up(current_pos, out->alignment);

  return 0;
}

// ---------------------------------------------------------------------------
// gguf_free
// ---------------------------------------------------------------------------
void gguf_free(gguf_file_t *f) {
  if (!f)
    return;

  if (f->kv) {
    for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
      free(f->kv[i].key);
      free_value(&f->kv[i].value);
    }
    free(f->kv);
  }

  if (f->tensors) {
    for (uint64_t i = 0; i < f->tensor_count; i++) {
      free(f->tensors[i].name);
    }
    free(f->tensors);
  }

  if (f->mmap_base)
    munmap(f->mmap_base, f->mmap_size);
  if (f->fd >= 0)
    close(f->fd);

  memset(f, 0, sizeof(*f));
  f->fd = -1;
}

// ---------------------------------------------------------------------------
// gguf_find — linear search for a metadata key
// ---------------------------------------------------------------------------
const gguf_value_t *gguf_find(const gguf_file_t *f, const char *key) {
  if (!f || !f->kv || !key)
    return NULL;
  for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
    if (strcmp(f->kv[i].key, key) == 0) {
      return &f->kv[i].value;
    }
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// gguf_type_name — map ggml_type to a short string
// ---------------------------------------------------------------------------
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
  default:
    snprintf(scratch, scratch_len, "UNKNOWN(%d)", (int)t);
    return scratch;
  }
}

// ---------------------------------------------------------------------------
// gguf_value_type_name
// ---------------------------------------------------------------------------
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
    snprintf(scratch, scratch_len, "UNKNOWN(%d)", (int)t);
    return scratch;
  }
}

// ---------------------------------------------------------------------------
// Helper: print a single metadata value (recursive for arrays)
// ---------------------------------------------------------------------------
static void print_value(const gguf_value_t *v, int indent) {
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
    printf("\"%s\"", v->v.v_str);
    break;
  case GGUF_TYPE_ARRAY:
  {
    const uint64_t shown = v->v.v_arr.len < 8 ? v->v.v_arr.len : 8;

    printf("[");
    for (uint64_t i = 0; i < shown; i++) {
      if (i > 0)
        printf(", ");
      print_value(&v->v.v_arr.items[i], indent + 1);
    }
    if (v->v.v_arr.len > shown)
      printf(", ... (%lu more)", v->v.v_arr.len - shown);
    printf("]");
    break;
  }
  default:
    printf("<%s>", gguf_value_type_name(v->type, scratch, sizeof(scratch)));
    break;
  }
}

// ---------------------------------------------------------------------------
// gguf_print_summary
// ---------------------------------------------------------------------------
void gguf_print_summary(const gguf_file_t *f, int max_tensors_shown) {
  if (!f)
    return;

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
  if (shown > f->tensor_count)
    shown = f->tensor_count;

  for (uint64_t i = 0; i < shown; i++) {
    const gguf_tensor_info_t *t = &f->tensors[i];
    char scratch[32];
    const char *tname = gguf_type_name(t->type, scratch, sizeof(scratch));

    printf("%s: type=%s, dims=[", t->name, tname);
    for (uint32_t d = 0; d < t->n_dims; d++) {
      printf("%lu", t->dims[d]);
      if (d < t->n_dims - 1)
        printf(", ");
    }
    printf("], offset=%lu\n", t->offset);
  }

  if (shown < f->tensor_count) {
    printf("... (%lu more tensors)\n", f->tensor_count - shown);
  }
}
