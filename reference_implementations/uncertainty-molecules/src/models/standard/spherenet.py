from dig.threedgraph.method import SphereNet as OriginalSphereNet
from torch_geometric.data import Data


class SphereNet(OriginalSphereNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z, pos, batch=None):
        batch_data = Data(z=z, pos=pos, batch=batch)
        return super(SphereNet, self).forward(batch_data)
