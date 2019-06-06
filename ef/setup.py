import setuptools

setuptools.setup(
    name='ef',
    version='0.1',
    install_requires=["numpy", "h5py", "scikit-learn"],
    packages=setuptools.find_packages(),
)