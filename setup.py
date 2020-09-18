from distutils.core import setup
setup(
  name = 'nextoff',         
  packages = ['nextoff', 'nextoff.data', 'nextoff.models'],   # Chose the same as "name"
  version = '0.1.2.2',      # Always update!
  license='MIT',        
  description = 'A package to experiment and tune keras hyperparameters',   # Give a short description about your library
  author = 'Asapanna Rakesh',                   
  author_email = 'rakeshark22@gmail.com',      
  url = 'https://github.com/rakesh4real/nextoff',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/rakesh4real/nextoff/archive/v0.1.2.2.tar.gz',  # link pasted from release
  keywords = ['Keras', 'hyperparameters', 'hyperparameters tuning'],   
  install_requires=[        
          'keras',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)