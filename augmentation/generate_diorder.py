import numpy as np
from scipy.interpolate import interp1d
import mayavi.mlab as mlab

class DisorderGenerator(object):
    def __init__(self, num_points=32, noise_level=0.1):
        self.num_points = num_points
        self.noise_level = noise_level
        self.q_3d = np.linspace(0, 1, num_points)
        self.Qx, self.Qy, self.Qz = np.meshgrid(self.q_3d, self.q_3d, self.q_3d, indexing='ij')
        self.R = np.sqrt(self.Qx**2 + self.Qy**2 + self.Qz**2)
        self.morphology = None

    def generate_morphology(self, bump_center=0, sigma=0.08):
        q = np.linspace(0, 1, 100)
        scattering = np.exp(-((q-bump_center)**2) / (2*sigma**2))

        # plt.plot(q, scattering)
        # plt.title('1D Scattering Pattern')
        # plt.xlabel('q')
        # plt.ylabel('Intensity')
        # plt.show()

        f = interp1d(q, scattering, fill_value="extrapolate")
        scattering_3d = f(self.R)

        scattering_3d_noisy = scattering_3d * (1 + self.noise_level * np.random.normal(size=scattering_3d.shape))
        random_phases = np.random.uniform(-np.pi, np.pi, scattering_3d.shape)
        complex_spectrum = scattering_3d_noisy * np.exp(1j * random_phases)

        self.morphology = np.fft.ifftn(complex_spectrum).real
        self.morphology -= self.morphology.min()
        self.morphology /= self.morphology.max()

    def visualize_morphology(self):
        if self.morphology is not None:
            mlab.figure(bgcolor=(1, 1, 1))
            mlab.contour3d(self.morphology, contours=[0.5], opacity=0.5)
            mlab.axes()
            mlab.show()
        else:
            print("Morphology not generated yet.")

    def save_morphology(self, filename="./disorder.rf"):
        if self.morphology is not None:
            np.savetxt(filename, self.morphology.reshape(-1, 1), fmt="%.6f")
        else:
            print("Morphology not generated yet.")

if __name__ == "__main__":
    generator = DisorderGenerator()
    generator.generate_morphology(bump_center=0, sigma=np.random.uniform(0.08, 0.08))

    generator.visualize_morphology()
    generator.save_morphology()
