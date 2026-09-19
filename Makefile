CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -g -std=c11 -Iinclude
LDFLAGS  =
BUILD    = build
TARGET   = minilmc

LIB_SRC  = src/gguf.c src/ggml_types.c
APP_SRC  = src/main.c
SRC      = $(LIB_SRC) $(APP_SRC)
OBJ      = $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))
TEST_TYPES_BIN = $(BUILD)/test_ggml_types
TEST_GGUF_BIN  = $(BUILD)/test_gguf_loader

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%.o: src/%.c include/gguf.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(TARGET) /tmp/gguf_test_*.gguf

run: $(TARGET)
	./$(TARGET) show $(MODEL)

$(TEST_TYPES_BIN): tests/test_ggml_types.c src/ggml_types.c include/ggml_types.h | $(BUILD)
	$(CC) $(CFLAGS) $< src/ggml_types.c -o $@

$(TEST_GGUF_BIN): tests/test_gguf_loader.c $(LIB_SRC) include/gguf.h include/ggml_types.h | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB_SRC) -o $@

test: $(TEST_TYPES_BIN) $(TEST_GGUF_BIN)
	$(TEST_TYPES_BIN)
	$(TEST_GGUF_BIN)

.PHONY: all clean run test
