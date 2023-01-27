import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setuptools.setup(
    name="simple-diffusion",
    packages=setuptools.find_packages(),
    version="0.0.1",
    license="MIT",
    description="simple diffusion: End-to-end diffusion for high resolution images in PyTorch and JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = "Suraj Patil",
    author_email = "surajp815@gmail.com",
    url = "https://github.com/patil-suraj/simple-diffusion",
    keywords = [
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'text-to-image'
    ],
    install_requires=[
        "accelerate",
        "diffusers",
        "pillow",
        "sentencepiece",
        "torch>=1.6",
        "transformers",
        "torch>=1.6",
        "torchvision",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)