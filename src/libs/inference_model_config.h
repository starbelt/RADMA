#ifndef INFERENCE_MODEL_CONFIG_H_
#define INFERENCE_MODEL_CONFIG_H_

// All 36 compiled models (Grid search: 6 alphas × 6 depths)
// Alpha values:  0.25, 0.5, 0.75, 1.0, 1.25, 1.5
// Depth multipliers: 2, 4, 6, 8, 10, 12

#define MODEL_BASE_PATH "data/models/coral_target/"

#define MODEL_PATH_A025_D02  MODEL_BASE_PATH "Grid_A0.25_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A025_D04  MODEL_BASE_PATH "Grid_A0.25_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A025_D06  MODEL_BASE_PATH "Grid_A0.25_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A025_D08  MODEL_BASE_PATH "Grid_A0.25_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A025_D10  MODEL_BASE_PATH "Grid_A0.25_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A025_D12  MODEL_BASE_PATH "Grid_A0.25_D12_quant_edgetpu.tflite"

#define MODEL_PATH_A050_D02  MODEL_BASE_PATH "Grid_A0.5_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A050_D04  MODEL_BASE_PATH "Grid_A0.5_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A050_D06  MODEL_BASE_PATH "Grid_A0.5_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A050_D08  MODEL_BASE_PATH "Grid_A0.5_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A050_D10  MODEL_BASE_PATH "Grid_A0.5_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A050_D12  MODEL_BASE_PATH "Grid_A0.5_D12_quant_edgetpu.tflite"

#define MODEL_PATH_A075_D02  MODEL_BASE_PATH "Grid_A0.75_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A075_D04  MODEL_BASE_PATH "Grid_A0.75_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A075_D06  MODEL_BASE_PATH "Grid_A0.75_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A075_D08  MODEL_BASE_PATH "Grid_A0.75_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A075_D10  MODEL_BASE_PATH "Grid_A0.75_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A075_D12  MODEL_BASE_PATH "Grid_A0.75_D12_quant_edgetpu.tflite"

#define MODEL_PATH_A100_D02  MODEL_BASE_PATH "Grid_A1.0_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A100_D04  MODEL_BASE_PATH "Grid_A1.0_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A100_D06  MODEL_BASE_PATH "Grid_A1.0_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A100_D08  MODEL_BASE_PATH "Grid_A1.0_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A100_D10  MODEL_BASE_PATH "Grid_A1.0_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A100_D12  MODEL_BASE_PATH "Grid_A1.0_D12_quant_edgetpu.tflite"

#define MODEL_PATH_A125_D02  MODEL_BASE_PATH "Grid_A1.25_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A125_D04  MODEL_BASE_PATH "Grid_A1.25_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A125_D06  MODEL_BASE_PATH "Grid_A1.25_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A125_D08  MODEL_BASE_PATH "Grid_A1.25_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A125_D10  MODEL_BASE_PATH "Grid_A1.25_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A125_D12  MODEL_BASE_PATH "Grid_A1.25_D12_quant_edgetpu.tflite"

#define MODEL_PATH_A150_D02  MODEL_BASE_PATH "Grid_A1.5_D02_quant_edgetpu.tflite"
#define MODEL_PATH_A150_D04  MODEL_BASE_PATH "Grid_A1.5_D04_quant_edgetpu.tflite"
#define MODEL_PATH_A150_D06  MODEL_BASE_PATH "Grid_A1.5_D06_quant_edgetpu.tflite"
#define MODEL_PATH_A150_D08  MODEL_BASE_PATH "Grid_A1.5_D08_quant_edgetpu.tflite"
#define MODEL_PATH_A150_D10  MODEL_BASE_PATH "Grid_A1.5_D10_quant_edgetpu.tflite"
#define MODEL_PATH_A150_D12  MODEL_BASE_PATH "Grid_A1.5_D12_quant_edgetpu.tflite"

// Ordered table of all 36 model paths for sequential iteration
// #define ALL_MODEL_PATHS \
//     MODEL_PATH_A025_D02, MODEL_PATH_A025_D04, MODEL_PATH_A025_D06, \
//     MODEL_PATH_A025_D08, MODEL_PATH_A025_D10, MODEL_PATH_A025_D12, \
//     MODEL_PATH_A050_D02, MODEL_PATH_A050_D04, MODEL_PATH_A050_D06, \
//     MODEL_PATH_A050_D08, MODEL_PATH_A050_D10, MODEL_PATH_A050_D12, \
//     MODEL_PATH_A075_D02, MODEL_PATH_A075_D04, MODEL_PATH_A075_D06, \
//     MODEL_PATH_A075_D08, MODEL_PATH_A075_D10, MODEL_PATH_A075_D12, \
//     MODEL_PATH_A100_D02, MODEL_PATH_A100_D04, MODEL_PATH_A100_D06, \
//     MODEL_PATH_A100_D08, MODEL_PATH_A100_D10, MODEL_PATH_A100_D12, \
//     MODEL_PATH_A125_D02, MODEL_PATH_A125_D04, MODEL_PATH_A125_D06, \
//     MODEL_PATH_A125_D08, MODEL_PATH_A125_D10, MODEL_PATH_A125_D12, \
//     MODEL_PATH_A150_D02, MODEL_PATH_A150_D04, MODEL_PATH_A150_D06, \
//     MODEL_PATH_A150_D08, MODEL_PATH_A150_D10, MODEL_PATH_A150_D12

// #define NUM_MODELS 36

#endif // INFERENCE_MODEL_CONFIG_H_