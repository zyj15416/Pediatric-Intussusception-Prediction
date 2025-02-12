import imgaug.augmenters as iaa
from config import params as cfg

class AdvancedAugmentation:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=(-30, 30),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1)
            ),
            iaa.MultiplyBrightness((0.8, 1.2)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.GaussianBlur(sigma=cfg.AUGMENTATION_PARAMS['blur_sigma']),
            iaa.AdditiveGaussianNoise(scale=cfg.AUGMENTATION_PARAMS['noise_scale']*255),
            iaa.MultiplyHueAndSaturation((-0.2, 0.2)),
            iaa.ElasticTransformation(alpha=50, sigma=5)
        ])
        
    def __call__(self, img):
        return self.seq(image=img)
