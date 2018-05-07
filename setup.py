from setuptools import setup
from setuptools import find_packages

setup(
   name='gym-trading',
   version='0.1',
   description='Trading environments for commodities and equities with historical data support',
   author='Marin Vlastelica',
   author_email=None,
   packages=find_packages(),
   install_requires=['gym', 'pandas', 'numpy'],
)
