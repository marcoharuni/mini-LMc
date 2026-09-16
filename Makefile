CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -g -std=c11 -Iinclude
LDFLAGS  =
BUILD    = build
TARGET   = minilmc

SRC      = $(wildcard src/*.c)
OBJ      = $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) $(TARGET)

run: $(TARGET)
	./$(TARGET) show $(MODEL)

.PHONY: all clean run