from setuptools import setup, find_packages 
from typing import List
def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='my_package',
    version='0.0.1',
    author='DareBadejo',
    author_email='ooa.badejo@gmail.com', 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
