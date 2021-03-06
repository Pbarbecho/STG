from setuptools import setup, find_packages
from os import path


local = path.abspath(path.dirname(__file__))

# README file
with open(path.join(local, 'README.md'), encoding='utf-8') as file:
    readme_file = file.read()

with open(path.join(local, 'LICENSE'), encoding='utf-8') as file:
    license_file = file.read()
    

setup(
    name='stg',
    url='https://github.com/Pbarbecho/STG.git',
    version='1.0',
    description='SUMO traffic generator',
    long_description=readme_file,
    long_description_content_type='text/markdown',
    py_modules=['stg'],
    author='Pablo Barbecho',
    author_email='pablo.barbecho@upc.edu',
    keywords='SUMO Traffic demand generation',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=['click', 'shutil','psutil','tqdm','sumolib', 'matplotlib', 'numpy ', 'pandas',
                      'pathlib', 'uuid', 'scipy', 'pivottablejs', 'seaborn', 'joblib', 'ipython', 'dask[dataframe]'],
    entry_points={
        'console_scripts': [
            'stg=stg:cli',
        ],
    },
    license="GNU GPL v2",
)
