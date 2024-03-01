from setuptools import setup, find_packages

setup(name='uncertainty-molecules',
      version='0.1',
      description='Uncertainty Molecules',
      author='Bertrand Charpentier, Tom Wollschlager',
      author_email='charpent@in.tum.de, tom.wollschlaeger@tum.de',
      packages=['src'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm',
                        'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2'],
      zip_safe=False)
