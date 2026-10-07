from setuptools import setup, find_packages

setup(
    name='pyphoon3',
    version='3.0.0',
    description='Dataloader for the Digital Typhoon Dataset (Kitamoto Lab)',
    url='https://github.com/kitamoto-lab/pyphoon3',
    author='Asanobu Kitamoto, Jared Hwang, Tong Ngoc Anh, Victor Le Louarn',
    author_email='kitamoto@nii.ac.jp',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'pandas',
        'h5py',
        'psutil',
    ],
    zip_safe=False
)
