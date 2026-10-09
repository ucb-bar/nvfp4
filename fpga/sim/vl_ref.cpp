// Cycle-accurate reference for one dut_wrap: stdin has one hex frame (IN_BYTES, byte 0 first)
// per line, stdout gets the OUT_BYTES result frame the FPGA harness would capture.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "Vdut_wrap.h"
#include "verilated.h"

template <typename T> static void put(T& sig, const uint8_t* b, int n) {
    sig = 0;
    for (int i = n - 1; i >= 0; --i) sig = (sig << 8) | b[i];
}
template <std::size_t W> static void put(VlWide<W>& sig, const uint8_t* b, int n) {
    for (std::size_t w = 0; w < W; ++w) sig[w] = 0;
    for (int i = 0; i < n; ++i) sig[i / 4] |= uint32_t(b[i]) << (8 * (i % 4));
}
template <typename T> static void get(const T& sig, uint8_t* b, int n) {
    for (int i = 0; i < n; ++i) b[i] = (uint64_t(sig) >> (8 * i)) & 0xff;
}
template <std::size_t W> static void get(const VlWide<W>& sig, uint8_t* b, int n) {
    for (int i = 0; i < n; ++i) b[i] = (sig[i / 4] >> (8 * (i % 4))) & 0xff;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vdut_wrap top;
    auto tick = [&] {
        top.clk = 0;
        top.eval();
        top.clk = 1;
        top.eval();
    };

    top.start = 0;
    top.rst = 1;
    for (int i = 0; i < 8; ++i) tick();
    top.rst = 0;
    tick();

    uint8_t in[IN_BYTES], out[OUT_BYTES];
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() < 2 * IN_BYTES) continue;
        for (int i = 0; i < IN_BYTES; ++i) in[i] = std::stoi(line.substr(2 * i, 2), nullptr, 16);
        put(top.in_data, in, IN_BYTES);
        top.start = 1;
        tick();
        top.start = 0;
        int cycles = 0;
        while (!top.done && cycles < 1000) {
            tick();
            ++cycles;
        }
        get(top.out_data, out, OUT_BYTES);
        for (int i = 0; i < OUT_BYTES; ++i) printf("%02x", out[i]);
        printf(" %d\n", top.done ? cycles + 1 : -1);
    }
    return 0;
}
