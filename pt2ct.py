import torch
from py_utils.module import Model
import coremltools as ct

if __name__ == '__main__':
    model = Model.load_from_checkpoint(
        "best_model/birds-epoch=02-val_loss=0.36.ckpt")
    X = torch.rand(1, 3, 112, 112)

    image_input = ct.ImageType(name="input_1",
                               shape=X.shape,
                               scale=1/255.0)

    model.to_torchscript(file_path="best_model/model_trace.pt", method='trace',
                         example_inputs=X)

    traced_model = torch.jit.trace(torch.load('best_model/model_trace.pt'), X)

    model = ct.convert(
        traced_model,
        inputs=[image_input],
        compute_units=ct.ComputeUnit.ALL,
    )
    model.save("best_model/bird.mlmodel")
