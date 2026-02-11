from setuptools import setup

setup(
    name='CFD',
    version='0.1',
    description='Library for CFD simulation',
    author='Tae Young Choi',
    author_email='tyul0529@naver.com',
    packages=['CFD'],
    install_requires=['torch'],
    license='MIT',
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
    ),
)
