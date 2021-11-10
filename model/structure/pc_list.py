


class PcList(object):
    def __init__(self, xyz_tensor, rgb_tensor=None,  batch_size=8):
        self.xyz = xyz_tensor
        self.rgb = rgb_tensor
        self.size = batch_size
        self.extra_field = {}

    def add_field(self, field_name, field_data):
        self.extra_field[field_name] = field_data

    def get_field(self, field_name):
        return self.extra_field[field_name]

