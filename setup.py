from setuptools import setup, find_packages

setup(
    name='openelo',
    version='0.0.2-2',
    author='Beniamin Condrea',
    author_email='benjamin.condrea@gmail.com',
    url='https://github.com/BeniaminC/Open-ELO',
    description='A set of elo systems written in Python.',
    long_description='file: README.md',
    keywords=['elo', 'rating system', 'mmr', 'competition'],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'nptyping', 'trueskill'],
    python_requires='>=3.12'
)