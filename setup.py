from distutils.core import setup
setup(
  name = 'nextoff',         # How you named your package folder (MyLib)
  packages = ['nextoff'],   # Chose the same as "name"
  version = '0.2.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A package to experiment and tune keras hyperparameters',   # Give a short description about your library
  author = 'Asapanna Rakesh',                   # Type in your name
  author_email = 'rakeshark22@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/rakesh4real/nextoff',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/rakesh4real/nextoff/archive/v0.2.1.tar.gz',  # link pasted from release
  keywords = ['Keras', 'hyperparameters', 'hyperparameters tuning'],   # Keywords that define your package best
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