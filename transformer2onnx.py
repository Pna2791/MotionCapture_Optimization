import torch.onnx
from torch.autograd import Variable


def load_model(name):
    from simple_transformer_with_state import TF_RNN_Past_State
    input_channels_imu = 6 * (9 + 3)
    output_channels = 18 * 6 + 3 + 20

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=True
    )
    model.load_state_dict(torch.load(name))
    return model

model_name = "output/model-with-dip9and10-cpu.pt"
model = load_model(model_name)

input_channels_imu = 6 * (9 + 3)+18
output_channels = 18 * 6 + 3 + 20

batch_size = 1
seq_len = 40
# dummy
dummy_input_imu = torch.randn(batch_size, seq_len, input_channels_imu)
dummy_input_s = torch.randn(batch_size, seq_len, output_channels)

model.eval()

# export onnx
onnx_path = 'output/your_model.onnx'
torch.onnx.export(model,
                  (dummy_input_imu, dummy_input_s),
                  onnx_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['imu_input', 's_input'],
                  output_names=['output'])

print(f"Mô hình đã được chuyển đổi thành {onnx_path}")