import numpy as np
import os
try:
    import matplotlib.pyplot as plt
except:
    print('Failed to import matplotlib')
import albumentations as A

from tools.image.castor import write_binary_file

class Phantom2DPetGenerator:

    def __init__(self, shape=(256, 256), voxel_size=(2,2,2), volume_activity=None):
        self.shape = shape
        self.voxel_size = voxel_size
        #
        if volume_activity is not None:
            self.volume_activity = volume_activity  # usually 1031.2922 kBq/ml
        else:
            self.volume_activity = None
            self.avg_volume_activity = self.calibrate(n_samples=100)

    def set_seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)

    def create_ellipse(self, center, axes):
        """
        Create binary mask with filled ellipse
        """
        x, y = np.linspace(-1, 1, self.shape[0]), np.linspace(-1, 1, self.shape[1])
        X, Y = np.meshgrid(x, y)
        x0, y0= center
        a, b = axes
        mask = ((X - x0) / a)**2 + ((Y - y0) / b)**2 <= 1
        return mask

    def create_body(self):
        """
        Create body mask.
        Body is the biggest component of the image.
        """
        a = np.clip(0.2*np.random.randn() + 1.6, 1.35, 1.85) / 2
        b = np.clip(0.2*np.random.randn() + 1, .75, 1.25) / 2
        self.body_a = a
        self.body_b = b
        return self.create_ellipse(center=(0, 0), axes=(a, b))

    def calibrate(self, n_samples=100):
        """Compute average volume activity in kBq/ml."""
        print('Computing average volume activity over', n_samples, 'samples...')
        volume_activities = []
        self.set_seed(42) # for consistent volume activity
        for _ in range(n_samples):
            obj, _ = self.create_phantom()
            body = obj > 0
            volume_activities.append(np.sum(obj) / (np.count_nonzero(body) * (self.voxel_size[0] * self.voxel_size[1] * self.voxel_size[2]) * 1e-3))  # kBq/mL
        self.avg_volume_activity = np.mean(volume_activities, dtype=np.float32)
        print('Average volume activity:', self.avg_volume_activity, 'kBq/mL')
        return self.avg_volume_activity

    def create_organ(self, body, axes, position=(0,0), size_ratio=0.8, attempt=1):
        """
        Create organ mask.
        Organs must be located in the body.
        They should not be overlapping too much with other organs as this is a sagittal section.
        Organs are translated and rotated randomly so that they fit in the body.
        """
        body_a, body_b = axes
        x0, y0 = position
        # Random rotation angle in radians
        theta = np.random.uniform(0, 2*np.pi)

        # Random position within the body ellipse
        r = np.random.uniform(0, 0.7)  # Keep organs away from body boundary
        phi = np.random.uniform(0, 2*np.pi)
        x0 = x0 + r * body_a * np.cos(phi)
        y0 = y0 + r * body_b * np.sin(phi)
        out_position = (x0, y0)

        # Random size that fits within body
        max_a = body_a * size_ratio  # Max organ size as fraction of body
        max_b = body_b * size_ratio
        a = np.clip(0.5*np.random.randn() + (max_a - 0.2)/2, 0.1, max_a)
        b = np.clip(0.5*np.random.randn() + (max_b - 0.2)/2, 0.1, max_b)        

        out = self.create_ellipse(center=(x0, y0), axes=(a, b))

        # Rotate the ellipse by theta around its center
        x = np.linspace(-1, 1, self.shape[0])
        y = np.linspace(-1, 1, self.shape[1])
        X, Y = np.meshgrid(x, y)

        # Translate to ellipse center
        Xc = X - x0
        Yc = Y - y0

        # Rotate coordinates by -theta to align with ellipse axes
        x_rot = Xc * np.cos(theta) + Yc * np.sin(theta)
        y_rot = -Xc * np.sin(theta) + Yc * np.cos(theta)

        # Recreate the ellipse mask with rotation applied
        out = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1

        max_attempts = 100
        if np.sum(out) != np.sum(np.logical_and(body, out)) and attempt < max_attempts:
            attempt += 1
            return self.create_organ(body, axes=(body_a, body_b), position=position, attempt=attempt)
        elif attempt == max_attempts:
            raise Exception('Max attempts reached in organ creation')
        else:
            return out, (a, b), out_position

    def create_tumour(self, organ, axes, position):
        """
        Fits a tumour within an organ.
        """
        return self.create_organ(organ, axes=axes, position=position, size_ratio=0.2)

    def get_activity_value(self, min, max):
        return np.random.uniform(min, max)

    def postprocess(self, img, seed=None):
        if seed is None:
            seed = np.random.randint(1,1e16)
        transform = A.Compose([
            A.GridDistortion(num_steps=5, p=1, distort_limit=.5),
            A.GridDistortion(num_steps=15, distort_limit=0.5, p=1)
        ], seed=seed)
        img = transform(image=img)['image']

        return img, seed

    def create_attenuation_map(self, obj, body_value, organ_values, body_att=0.096, air_att=0.029, organs_att=(0.05, 0.2)):
        """
        Creates random attenuation map for each connected region and the body.
        Gaussian filter is applied.

        body_att cm-1 is chosen for the body part.
        Organs with fewer activity than the body is expected to be air (like lungs). We choose air_att cm-1.
        Other organs attenuation value is randomly chosen within given range organs_att.
        No attenuation is chosen for tumours.
        """

        out = np.zeros_like(obj)
        # Body attenuation
        out = np.where(obj == body_value, body_att, out)
        # Air attenuation
        out = np.where(obj < body_value, air_att, out)
        # Organ attenuation
        for organ_value in organ_values:
            out = np.where(obj == organ_value, np.random.uniform(organs_att[0], organs_att[1]), out)
        # Fill the outside with 0
        out = np.where(obj == 0, 0, out)
        return out

    def create_phantom(self):
        """
        Generation function for body creation, organ embedding, tumour embedding.
        Returns object and attenuation map.
        """

        # Create body
        body = self.create_body()
        body_activity_min, body_activity_max = 5, 8
        body_value = self.get_activity_value(body_activity_min, body_activity_max)
        out = body * body_value
        #
        # Create organs
        organs = []
        organs_axes = []
        organs_position = []
        organs_mask = np.zeros_like(body)
        organ_values = []
        min_num_organs, max_num_organs = 3,6
        organ_activity_min, organ_activity_max = 1, 30
        num_organs = np.random.randint(min_num_organs, max_num_organs)
        for _ in range(num_organs):
            try:
                organ, axes, position = self.create_organ(np.logical_and(body, np.logical_not(organs_mask)), axes=(self.body_a, self.body_b))
            except:
                continue
            organs_axes.append(axes)
            organs_position.append(position)
            organs_mask = np.where(organ, 1., organs_mask)
            organs.append(organ)
            if len(organ_values) and any([i < body_value for i in organ_values]):
                organ_value = self.get_activity_value(min(min(organ_values), body_value), organ_activity_max)
            else:
                organ_value = self.get_activity_value(organ_activity_min, organ_activity_max)
            organ_values.append(organ_value)
            out = np.where(organ, organ_value, out)

        attenuation_map = self.create_attenuation_map(obj=out, body_value=body_value, organ_values=organ_values)

        # Create tumour(s)
        min_num_tumours, max_num_tumours = 1, 2
        tumour_activity_min, tumour_activity_max = 40, 60
        num_tumours = np.random.randint(min_num_tumours, max_num_tumours)
        for i, organ_value in enumerate(organ_values):
            if organ_value < body_value:
                organ_values.pop(i)
                organs.pop(i)
        if len(organs) > 0:
            organs_with_tumour_idx = np.random.choice(len(organs), replace=True, size=num_tumours)
            for organ_with_tumour_idx_ in organs_with_tumour_idx:
                organ_a, organ_b = organs_axes[organ_with_tumour_idx_]
                try:
                    tumour, _, _ = self.create_tumour(organs[organ_with_tumour_idx_], axes=(organ_a, organ_b), position=organs_position[organ_with_tumour_idx_])
                except:
                    continue
                out = np.where(tumour, self.get_activity_value(tumour_activity_min, tumour_activity_max), out)
        
        out, seed = self.postprocess(out, seed=None)
        attenuation_map, _ = self.postprocess(attenuation_map, seed=seed)

        # Scale activity values to match desired volume activity if specified
        if self.volume_activity is not None:
            body = out > 0
            current_volume_activity = np.sum(out) / (np.count_nonzero(body) * (self.voxel_size[0] * self.voxel_size[1] * self.voxel_size[2]) * 1e-3)  # kBq/mL
            scaling_factor = self.volume_activity / current_volume_activity
            out = out * scaling_factor

        return out, attenuation_map

    def save_file(self, img, dout):
        """Save file in the .img/.hdr format so that images can be used in castor"""

        metadata = {
            '!imaging modality': 'phantom',
            '!version of keys': 'CASToRv3.1',
            'CASToR version': '3.1',
            '!originating system': 'create_phantom',
            '!data offset in bytes': '0',
            '!name of data file': f'{os.path.basename(dout)}.img',
            'patient name': os.path.basename(dout),
            '!type of data': 'Static',
            '!total number of images': '1',
            'imagedata byte order': 'LITTLEENDIAN',
            '!study duration (sec)': '1',
            'number of dimensions': '3',
            '!matrix size [1]': str(img.shape[0]),
            '!matrix size [2]': str(img.shape[1]),
            '!matrix size [3]': '1',
            '!number format': 'short float',
            '!number of bytes per pixel': '4',
            'scaling factor (mm/pixel) [1]': str(self.voxel_size[0]),
            'scaling factor (mm/pixel) [2]': str(self.voxel_size[1]),
            'scaling factor (mm/pixel) [3]': str(self.voxel_size[2]),
            'first pixel offset (mm) [1]': '0',
            'first pixel offset (mm) [2]': '0',
            'first pixel offset (mm) [3]': '0',
            'data rescale offset': '0',
            'data rescale slope': '1',
            'quantification units': '1',
            '!image duration (sec)': '1',
            '!image start time (sec)': '0'
        }

        write_binary_file(dout, img, metadata=metadata, binary_extension='.img')

        return os.path.join(os.path.abspath('.'), dout)

    def run(self, dest_path):
        """
        Run phantom creation
        Returns object path and attenuation map path.
        """
        
        out, attenuation_map = self.create_phantom()

        # Save files
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        obj, att = out.astype(np.float32), attenuation_map.astype(np.float32)
        dout_obj = self.save_file(obj, os.path.join(dest_path, os.path.basename(dest_path)))
        dout_att = self.save_file(att, os.path.join(dest_path, os.path.basename(dest_path) + '_att'))

        return dout_obj, dout_att

if __name__ == '__main__':

    generator = Phantom2DPetGenerator(volume_activity=1e3)  # 1000 kBq/ml
    obj, att = generator.run(dest_path='./data')

    with open(obj + '.img', 'rb') as f:
        obj = np.fromfile(f, dtype=np.float32).reshape((256,256))
    with open(att + '.img', 'rb') as f:
        att = np.fromfile(f, dtype=np.float32).reshape((256,256))

    fig, ax = plt.subplots(1, 2)
    ax1 = ax[0].imshow(obj, cmap='gray_r', vmin=0, vmax=50)
    ax2 = ax[1].imshow(att, cmap='gray_r', vmin=0, vmax=0.2)
    plt.colorbar(ax1)
    plt.colorbar(ax2)
    plt.show()