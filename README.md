# mini-LMc

A from-scratch LLM inference engine 

## Requirements

- gcc
- make
- Linux (uses mmap, POSIX)

## Build

    make

Clean build artifacts:

    make clean

## Download a test model

    mkdir -p models
    wget -P models/ https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Any `.gguf` v3 file works — Llama, Qwen, Mistral, Gemma, Phi, MoE models, etc.

## Usage

### Show all model info

    ./minilmc show models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

### Show only the first N tensors

    ./minilmc show models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 20

### Show only metadata (first 60 lines)

    ./minilmc show models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | head -60

### Show only the tensor table

    ./minilmc show models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | sed -n '/--- Tensors ---/,$p'

## Inspect the raw GGUF binary

### First 40 bytes (header + metadata start)

    xxd models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | head -40

You'll see:

    00000000: 4747 5546 0300 0000 c900 0000 0000 0000  GGUF............
    00000010: 1700 0000 0000 0000 1400 0000 0000 0000  ................
    00000020: 6765 6e65 7261 6c2e 6172 6368 6974 6563  general.architec
    00000030: 7475 7265 0800 0000 0500 0000 0000 0000  ture............
    00000040: 6c6c 616d 610c 0000 0000 0000 0067 656e  llama........gen

- `4747 5546` = "GGUF" magic
- `0300 0000` = version 3
- `c900 0000 0000 0000` = 201 tensors
- `1700 0000 0000 0000` = 23 metadata keys

### Jump to tensor data section

    xxd -s 0x1a1580 models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | head -20

(Replace `0x1a1580` with the "Tensor data at" offset printed by `./minilmc show`.)

### Browse the whole file

    xxd models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | less

### File size

    ls -lh models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

## Project Layout

    include/    Public headers (gguf.h)
    src/        Source files (gguf.c, main.c)
    models/     Test .gguf files (gitignored)
    build/      Object files (gitignored)
    docs/       Design notes
    scripts/    Helper scripts
    tests/      Test files and reference dumps

## Requirements

- gcc
- make
- Linux (uses mmap, POSIX)

## License

TBD
