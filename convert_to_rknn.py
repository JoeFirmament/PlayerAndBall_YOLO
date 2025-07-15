import sys
from rknn.api import RKNN

def convert_to_rknn():
    model_path = 'models/PlayerAndBasketball_rknn_ready.onnx'
    platform = 'rk3588'
    output_path = 'models/PlayerAndBasketball_fp16.rknn'

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model with float16 (no quantization)
    print('--> Building model with float16')
    ret = rknn.build(do_quantization=False)  # 不进行量化，使用float16
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
    print(f'Successfully converted to: {output_path}')

if __name__ == '__main__':
    convert_to_rknn() 