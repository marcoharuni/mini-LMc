#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s show <model.gguf> [max_tensors]\n"
          "\n"
          "  show       Print model metadata and tensor table.\n"
          "             max_tensors: optional limit (-1 = all).\n",
          prog);
}

int main(int argc, char **argv) {
  if (argc < 3 || strcmp(argv[1], "show") != 0) {
    usage(argv[0]);
    return 1;
  }

  const char *path = argv[2];
  int max_tensors = -1;
  if (argc >= 4) {
    max_tensors = atoi(argv[3]);
  }

  gguf_file_t f;
  if (gguf_load(path, &f) != 0) {
    fprintf(stderr, "Failed to load %s\n", path);
    return 1;
  }

  gguf_print_summary(&f, max_tensors);

  gguf_free(&f);
  return 0;
}