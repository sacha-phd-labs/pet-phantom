import numpy as np
import shutil

from tools.image.castor import read_castor_binary_file

from phantom_simulation.object_simulator import Phantom2DPetGenerator

class TestObjectSimulatorSeed:

    def generate_obj_att(self, seed=None, **kwargs):
        gen = Phantom2DPetGenerator(**kwargs)
        gen.set_seed(seed)
        obj_path, att_path = gen.run(dest_path='./tmp_object')
        obj = read_castor_binary_file(obj_path + '.hdr')
        att = read_castor_binary_file(att_path + '.hdr')
        shutil.rmtree('./tmp_object')
        return obj, att

    def test_object_simulator_seed(self):

        # Run
        obj1, att1 = self.generate_obj_att(seed=42, volume_activity=1e3)
        obj2, att2 = self.generate_obj_att(seed=42, volume_activity=1e3)

        assert np.all(np.equal(obj1, obj2)), "Different objects with same seed"
        assert np.all(np.equal(att1, att2)), "Different attenuation maps with same seed"

    def test_volume_activity(self):
        """Test that a calibrated generator is consistent"""
        #
        volume_activity_target = 1e3
        obj, _ = self.generate_obj_att(seed=None, volume_activity=volume_activity_target, voxel_size=(2.0, 2.0, 2.0))
        #
        volume_activity_obj = np.sum(obj) / (np.count_nonzero(obj) * (2.0)**3 * 1e-3)  # kBq/mL
        assert np.abs(volume_activity_obj - volume_activity_target) < 1e-2
        
if __name__ == "__main__":
    test = TestObjectSimulatorSeed()
    test.test_object_simulator_seed()
    test.test_volume_activity()