from setuptools import setup

name = 'bkntis'
packages = ['bkntis', 'bkntis/util', 'bkntis/graph']
keywords = ['R&D Investment', "R&D Funding"]
description = 'Python package for analyzing and visualizing the R&D investment in South Korea.'
install_requires = ['pandas', 'plotly','matplotlib','XlsxWriter','PyPDF2', 'scipy', 'kaleido','openpyxl']

if __name__ == "__main__":
    setup(
        name=name,
        version='0.0.2',
        packages=packages,
        url='https://github.com/skyboong/package_bkntis',
        license='GNU GENERAL PUBLIC LICENSE V3',
        author='B K Choi',
        author_email='stsboongkee@gmail.com',
        description=description,
        keywords=keywords,
        install_requires=install_requires,
        python_requires=">=3.10",
    )

