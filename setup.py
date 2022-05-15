from setuptools import setup, find_packages

exec(open("models/__version__.py").read())

setup(
    name="gans",
    version=__version__,
    author="Caibin Sheng",
    author_email="shengcaibin@gmail.com",
    description="GANs",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    install_requires=[
        "torch>=1.10.0",
        "pandas>=1.3.4",
        "torchvision>=0.9.0",
        "seaborn>=0.11.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific :: Artificial Intelligence",
    ],
    zip_safe=False,
)