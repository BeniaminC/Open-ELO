from setuptools import setup, find_packages

setup(
    name='openelo',
    version='0.0.1',
    author='Beniamin Condrea',
    author_email='benjamin.condrea@gmail.com',
    url='https://github.com/BeniaminC/Open-ELO',
    description='A set of elo systems written in Python.',
    packages=find_packages(),
    classifiers=[
        'Programming Langauges :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=['numpy', 'scipy', 'nptyping', 'trueskill']
)