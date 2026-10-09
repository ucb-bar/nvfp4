// Direct MMIO simulation of compute_test_top, driven by host/compute_sim.py.
//
// stdin, one command per line, hexadecimal arguments:
//   B <write> <address> <value> <mask>  one bus access; prints "<read data> <denied> <cycles>"
//   P <op> <x> <y>                      one fp32_alu operation; prints the result
//   F <x> <y>                           fp32_pkg add, mul, exp2, recip and max; prints the five results
//   Q                                   exit
#include "Vcompute_test_top.h"
#include "verilated.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>

static void tick(Vcompute_test_top& dut) {
  dut.clk = 0; dut.eval();
  dut.clk = 1; dut.eval();
  dut.clk = 0; dut.eval();
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vcompute_test_top dut;
  dut.rst = 1;
  dut.a_valid = 0;
  dut.probe_valid = 0;
  dut.d_ready = 1;
  for (int i = 0; i < 5; i++) tick(dut);
  dut.rst = 0;
  char command;
  uint32_t transactions = 0;
  while (std::cin >> command) {
    if (command == 'Q') break;
    if (command == 'F') {
      uint32_t a, b;
      std::cin >> std::hex >> a >> b;
      dut.probe_a = a; dut.probe_b = b; dut.eval();
      std::cout << std::hex << dut.probe_add << ' ' << dut.probe_mul << ' '
                << dut.probe_exp2 << ' ' << dut.probe_recip << ' ' << dut.probe_max << std::endl;
    } else if (command == 'P') {
      uint32_t op, a, b;
      std::cin >> std::hex >> op >> a >> b;
      int cycles = 0;
      while (!dut.probe_ready && cycles++ < 1000) tick(dut);
      dut.probe_operation = op; dut.probe_a = a; dut.probe_b = b; dut.probe_valid = 1;
      tick(dut); dut.probe_valid = 0; dut.eval();
      while (!dut.probe_complete && cycles++ < 1000) tick(dut);
      if (!dut.probe_complete) throw std::runtime_error("ALU timeout");
      std::cout << std::hex << dut.probe_result << std::endl;
      tick(dut);
    } else if (command == 'B') {
      uint32_t write, address, value, mask;
      std::cin >> std::hex >> write >> address >> value >> mask;
      dut.a[0] = value;
      dut.a[1] = (address << 4) | mask;
      dut.a[2] = (address >> 28) | ((write ? 0u : 4u) << 4);
      dut.a_valid = 1;
      dut.eval();
      int cycles = 0;
      while (!dut.a_ready && cycles++ < 100000) tick(dut);
      if (!dut.a_ready) throw std::runtime_error("MMIO request timeout");
      tick(dut);
      dut.a_valid = 0;
      dut.eval();
      while (!dut.d_valid && cycles++ < 100000) tick(dut);
      if (!dut.d_valid) throw std::runtime_error("MMIO response timeout");
      const uint64_t response = dut.d;
      // Exercise response backpressure while both compute devices can be running.
      if (++transactions % 17 == 0) {
        dut.d_ready = 0;
        for (int i = 0; i < 3; i++) {
          tick(dut);
          if (!dut.d_valid || dut.d != response) throw std::runtime_error("unstable stalled response");
        }
      }
      dut.d_ready = 1;
      tick(dut);
      std::cout << std::hex << uint32_t(response) << ' ' << ((response >> 32) & 1)
                << ' ' << cycles << std::endl;
    } else throw std::runtime_error("unknown command");
  }
}
