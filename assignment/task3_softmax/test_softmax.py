import pytest, torch
from solution import tl_softmax

@pytest.mark.parametrize('n,m', [(1,1),(3,7),(16,256),(17,513),(64,4096)])
def test_softmax(n,m):
    x=torch.randn((n,m),device='cuda'); out=tl_softmax(x,BLOCK_N=16,BLOCK_M=256)
    torch.testing.assert_close(out,torch.softmax(x,dim=1),atol=2e-3,rtol=2e-3)

def test_extreme():
    x=torch.tensor([[-1000.,0.,1000.,1.]],device='cuda'); out=tl_softmax(x,BLOCK_N=1,BLOCK_M=4)
    torch.testing.assert_close(out,torch.softmax(x,dim=1),atol=2e-3,rtol=2e-3)
