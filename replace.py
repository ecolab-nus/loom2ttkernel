import sys


mlir_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "tmp_output/tmp_mlir_files/kernel_ttkernel.mlir"
)
mlir_code = open(mlir_path, "r").read()
mlir_code = mlir_code.replace('ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>', '"ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"')
mlir_code = mlir_code.replace('ttkernel.experimental::convert_logical_x_to_translated', '"ttkernel.experimental::convert_logical_x_to_translated"')
mlir_code = mlir_code.replace('ttkernel.experimental::get_noc_multicast_addr', '"ttkernel.experimental::get_noc_multicast_addr"')
mlir_code = mlir_code.replace('ttkernel.experimental::convert_logical_y_to_translated', '"ttkernel.experimental::convert_logical_y_to_translated"')
open(mlir_path, "w").write(mlir_code)
