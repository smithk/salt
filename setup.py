from setuptools import setup

setup(name='salt-ml',
      version='0.1dev',
      description='Toolbox to suggest a machine learning technique and options best suited to a particular data set.',
      long_description=open('README.rst').read(),
      url='http://github.com/rogerberm/salt',
      author='Roger Bermudez-Chacon, Kevin Smith, Peter Horvath',
      author_email='beroger@student.ethz.ch',
      license='[pending to define]',
      packages=['salt', 'salt.data', 'salt.evaluate', 'salt.gui', 'salt.IO', 'salt.jobs',
                'salt.learn', 'salt.optimize', 'salt.options', 'salt.parameters', 'salt.sample',
                'salt.suggest', 'salt.utils'],
      data_files=[('salt/gui/images', ['salt/gui/images/*.gif']), ],
      platforms='any',
      zip_safe=False,
      entry_points="""
      [console_scripts]
      salt = salt.main:main
      """,
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License ::Other/Proprietary License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development', ]
      )
