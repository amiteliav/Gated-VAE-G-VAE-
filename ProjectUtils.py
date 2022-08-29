import torch


def choose_cuda(cuda_num):
    if cuda_num=="cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if 0 <= cuda_num <= device_count - 1:  # devieces starts from '0'
            device = torch.device(f"cuda:{cuda_num}")
        else:
            print(f"Cuda Num:{cuda_num} NOT found, choosing cuda:0")
            device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")

    print("*******************************************")
    print(f" ****** running on device: {device} ******")
    print("*******************************************")
    return device