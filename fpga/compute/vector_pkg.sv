// Operations of the vector device. The value is the op field of the command.
package vector_pkg;
  typedef enum logic [3:0] {
    V_ADD, V_SUB, V_MUL, V_SCALE, V_AXPY, V_MAX, V_SUM,
    V_EXP2, V_RECIP, V_SOFTMAX, V_NORMALIZE, V_RSQRT, V_SILU, V_ABS,
    V_QUANT  // NVFP4 block quantization; flash_vector runs it, flash_vector_unit does not
  } vector_op_e;
endpackage
