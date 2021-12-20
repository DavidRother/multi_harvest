from setuptools import setup, find_packages

setup(name='gathering_zoo',
      version='0.0.1',
      description='Gathering Environment',
      author='David Rother',
      email='rother.work@gmail.com',
      packages=find_packages() + [""],
      install_requires=[
            'gym==0.19.0',
            'numpy>=1.21.2',
            'pygame==2.0.1',
            'pettingzoo==1.11.2',
            'pillow==8.4.0'
      ]
      )
