from setuptools import setup, find_packages

setup(
    name='sfq_sim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'qutip'
    ],
    author='Leon Guerrero',
    author_email='leonmario.guerrero@gmail.com',
    description='simulating SFQ control of superconducting qubits',
    url='https://github.com/LeonGuerrero-98/sfq_sim', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)