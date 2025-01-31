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
        'qutip >= 5.0'
    ],
    author='Leon M. Guerrero',
    author_email='leonmario.guerrero@gmail.com',
    description='simulating SFQ control of superconducting qubits',
    url='https://github.com/LeonGuerrero-98/sfq_sim', 
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords='superconducting quantum circuit simulation',
    python_requires='>=3.10',
    )