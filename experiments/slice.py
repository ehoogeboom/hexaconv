import chainer


class Slice(chainer.Function):

    def __init__(self, slices):
        self.slices = slices

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == np.float32)

    def forward(self, inputs):
        x = inputs[0]
        return x[self.slices],

    def backward(self, inputs, grad_outputs):
        x = inputs[0]
        xp = cuda.get_array_module(x)
        grad_input = xp.zeros_like(x)
        grad_input[self.slices] = grad_outputs[0]
        return grad_input,
