CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -g -std=c11 -Iinclude
LDFLAGS  =
BUILD    = build
TARGET   = minilmc

LIB_SRC  = src/gguf.c src/ggml_types.c
APP_SRC  = src/main.c
SRC      = $(LIB_SRC) $(APP_SRC)
OBJ      = $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))
TEST_BIN = $(BUILD)/test_ggml_types

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%.o: src/%.c include/gguf.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(TARGET)

run: $(TARGET)
	./$(TARGET) show $(MODEL)

$(TEST_BIN): tests/test_ggml_types.c src/ggml_types.c include/ggml_types.h | $(BUILD)
	$(CC) $(CFLAGS) $< src/ggml_types.c -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

.PHONY: all clean run test
