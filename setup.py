from setuptools import setup

setup(
  name = 'stVGP',         
  packages = ['stVGP'],  
  version = '0.0.2',      
  license= 'MIT',        
  description = 'A variational spatiotemporal Gaussian process framework designed to integrate multi-modal, ' \
  'multi-slice spatial transcriptomics (ST) data for coherent 3D tissue reconstruction',   
  author = 'Zedong Wang',                   
  author_email = 'wangzedong23@mails.ucas.ac.cn',     
  url = 'https://github.com/wzdrgi/stVGP',   
  keywords = ["spatial transcriptomics", 'domain ', '3D reconstruction'],   
  install_requires=[           
          'scipy',  
          'numpy',
          'pandas',
          'scanpy',
          'scikit-learn',
          'anndata',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',   
    'Programming Language :: Python :: 3.5', 
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
  ],
)
