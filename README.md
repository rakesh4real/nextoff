# nextoff

Inspired by [Blog](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)

Generate `*.tar.gz` in `dist/` dir
```
$ python3 setup.py sdist      
 ```

Install twine and upload (requires uname and pass)
```
$ pip3 install twine
$ twine upload dist/*
```