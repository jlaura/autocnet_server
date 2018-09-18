from setuptools import setup, find_packages
setup(
    name='autocnet_server',
    version='0.1.5',
    long_description='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    scripts=['bin/acn_submit',]
)
