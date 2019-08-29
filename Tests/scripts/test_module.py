import torch

class Test(torch.jit.ScriptModule):
    def __init__(self):
        super(Test, self).__init__()

    @torch.jit.script_method
    def forward(self, input):
        return None

    @torch.jit.script_method
    def eqBool(self, input):
        # type: (bool) -> bool
        return input

    @torch.jit.script_method
    def eqInt(self, input):
        # type: (int) -> int
        return input

    @torch.jit.script_method
    def eqFloat(self, input):
        # type: (float) -> float
        return input

    @torch.jit.script_method
    def eqTensor(self, input):
        # type: (Tensor) -> Tensor
        return torch.add(input, input)

    @torch.jit.script_method
    def eqBoolList(self, input): 
        # type: (List[bool]) -> List[bool]
        return input

    @torch.jit.script_method
    def eqIntList(self, input): 
        # type: (List[int]) -> List[int]
        return input
    
    @torch.jit.script_method
    def eqFloatList(self, input):
        # type: (List[float]) -> List[float]
        return input
    
    @torch.jit.script_method
    def eqTensorList(self, input):
        # type: (List[Tensor]) -> List[Tensor]
        return input

OUTPUT_DIR = '../models/test.pt'
print('-' * 80)
traced_model = Test()
print(traced_model.graph)
print(traced_model.code)
traced_model.save(OUTPUT_DIR)
print("Saved to " + OUTPUT_DIR)
print('=' * 80)