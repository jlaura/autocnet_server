from setuptools import setup, find_packages
setup(
    name='autocnet_server',
    version='0.0.1',
    long_description='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_requires=[],
    scripts=['bin/acn_create_network', 'bin/acn_estimate_fundamental',
             'bin/acn_extract_features', 'bin/acn_generate_mosaic',
             'bin/acn_ring_match']
)
